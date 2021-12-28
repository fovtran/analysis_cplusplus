#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
# :Licence:   This work may be distributed and/or modified under the
#             conditions of the `LaTeX Project Public License`_,
#             either version 1.3 of this license or (at your option)
#             any later version.
# :Id: $Id:  $

# ===========================================================================

"""Parse and compare LyX's unicodedefs LaTeX <-> Unicode mappings"""

import os, sys, shlex, unicodedata
import parse_unimathsymbols

unidata = parse_unimathsymbols.read_data()

# relevant math preable features present in LyX 2.0:
lyx20features = ['', 'amssymb', 'esint', 'txfonts', 'wasysym']

infile = file('/usr/local/src/lyx/lib/unicodesymbols', 'r')
# infile = file(os.path.expanduser('~/.lyx-svn/unicodesymbols'), 'r')
outfile = sys.stdout
# outfile = file(os.path.expanduser('~/.lyx-svn/unicodesymbols'), 'w')
# outfile = file('/usr/local/src/lyx/lib/unicodesymbols-mathpatch', 'w')

# UniLyXEntry
# ~~~~~~~~~~~~
#
# Data structure representing one character. Initialized with a string
# in the format of LyX's `unicodesymbols` file:
#
# ``ucs4 "textcommand"  "textpreamble" "flags" "mathcommand" "mathpreamble" # comment``
#
# * textcommand and textpreamble are used if the symbol occurs in textmode.
# * mathcommand and mathpreamble are used if the symbol occurs in mathmode.
#
# Both mathcommand and mathpreamble are optional.
#
# * textpreamble and mathpreamble can either be a feature known
#   by the LaTeXFeatures class (e.g. tipa), or a LaTeX command 
#   (e.g. ``\\usepackage{bla}``).
#
# Known flags:
#
# - combining This is a combining char that will get combined with a base char
# - force     Always output replacement command
# - mathalpha This character is considered as a math variable in mathmode
#
# ::

class UniLyXEntry(object):
    # Data field, default, index after splitting at '"'
    codepoint = None
    textmacro = ''
    textpreamble = ''
    flags = ''
    mathmacro = None
    mathpreamble = None
    comment = None

    def __init__(self, line):
        """Parse one `line` of LyX's unicodesymbols.txt"""
        tokens = shlex.split(line, comments=True)
        try:
            self.codepoint = int(tokens[0], 16)
            self.textmacro = tokens[1]
            self.textpreamble = tokens[2]
            self.flags = tokens[3]
            self.mathmacro = tokens[4]
            self.mathpreamble = tokens[5]
        except IndexError:
            pass
        try:
            self.comment = line.split('#', 1)[1].rstrip()
        except IndexError:
            pass

    def _field_repr(self, s):
        return '"%s"' % s.replace('\\', '\\\\').replace('"', '\\"')

    def __str__(self):
        """(Try to) return the reconstructed the source line.
        
        As the original file is hand-written, differences in
        spacing and syntactic variants are still possible.
        """
        # There are three optional parts: text, math, comment
        fields = []
        if self.codepoint:
            fields.append('0x%04x %-26s %s %s' %
                          (self.codepoint,
                           self._field_repr(self.textmacro),
                           self._field_repr(self.textpreamble),
                           self._field_repr(self.flags))
                         )
        fields += [self._field_repr(f) for f in (self.mathmacro,
                                                 self.mathpreamble)
                   if f is not None]
        if self.comment is not None:
            fields.append('#%s' % self.comment)
        return ' '.join(fields)


# Return a new entry for codepoint:
# 
# >>> print new_entry(126)

def new_entry(codepoint):
    """Return a new UniLyXEntry for Unicode char with `codepoint`

    Raise ValueError, is there is no Unicode character with that codepoint.
    """
    entry = UniLyXEntry('')
    entry.codepoint = codepoint
    entry.comment = ' %s %s' % (unichr(codepoint).encode('utf8'),
                                unicodedata.name(unichr(codepoint)))
    return entry

# Report differences
# ------------------

# An earlier variant of this function was used to "import" definitions
# in LyX ``unicodesymbols`` file into the database. Now it shows
# different definitions that hint to unresolved/ambiguous mappings::

def report_differences():
    """
    Write lines with different definitions in LyX and unimathsymbols.txt
    """
    print '--- ' + infile.name
    print '+++ generated from unimathsymbols database'
    
    for line in infile:
        lyxentry = UniLyXEntry(line)
        
        # skip, if there is no math part
        if not lyxentry.mathmacro:
            continue
        
# Skip "acknowledged" differences::

        # accented latin characters 
        # (no mathematical chars according to Unicode)
        if 'LATIN' in lyxentry.comment:
            continue
        
        # Sub- and Superscripts
        # (non-trivial to implement because of 'nesting')
        if ('SUBSCRIPT' in lyxentry.comment or 
            'SUPERSCRIPT' in lyxentry.comment):
            continue
        
        # Script vs. Calligraphic style:
        # if r'\mathscr' in lyxentry.mathmacro:
        #     continue
        
        # get matching unimathsymbols.txt entry:
        uentry = unidata.get(lyxentry.codepoint, 
                    parse_unimathsymbols.new_entry(lyxentry.codepoint))
        # skip if the macro is already present in equivalent form
        aliases = [ae.cmd for ae in uentry.related_commands('=')]
        if lyxentry.mathmacro in aliases + [uentry.cmd]:
            continue
        
# Report as 'unimathsymbols.txt' diff::

        # outfile.write('- %s\n' % uentry)
        # uentry.cmd = lyxentry.mathmacro
        # uentry.requirements = lyxentry.mathpreamble or ''
        # outfile.write('+ %s\n' % uentry)

# Report as LyX 'unicodesymbols' diff::

        outfile.write('- %s\n' % lyxentry)
        lyxentry.mathmacro = uentry.cmd
        lyxentry.mathpreamble = uentry.requirements
        outfile.write('+ %s\n' % lyxentry)

# Unify the file layout
# ---------------------

# As the original file is hand-written, differences in spacing and
# syntactic variants (e.g. ``\`` vs ``\\``) are still possible.

def reformat_unicodesymbols():
    for line in infile:
        lyxentry = UniLyXEntry(line)
        outfile.write(str(lyxentry) + '\n')

# Add missing math macros for feature set `features`
# --------------------------------------------------
# ::

def supported_mathmacro(codepoint, features):
    """
    Return new lyxentry for `codepoint`
    with data fields filled in from unimathsymbols
    if it is supported by the list of `features`.
    """
    try:
        uentry = unidata[codepoint]
    except KeyError:
        return None
    # TODO: support for combining characters, accents and radicals:
    if uentry.category in ('mathaccent', 'mathradical',
                          'mathover', 'mathunder'):
        return None

    for feature in features:
        cmd = uentry.supported_cmd([feature])
        if cmd:
            entry = new_entry(codepoint)
            # add 'mathalpha' flag for italicised letters
            if (uentry.category == 'mathalpha' and
                not cmd.startswith(r'\math') and
                uentry.supported_cmd([feature, 'literal']) != cmd):
                entry.flags = 'mathalpha'
            entry.mathmacro = cmd
            entry.mathpreamble = feature
            return entry
    return None

def list_mathpreambles():
    mathpreambles = {}
    for line in infile:
        mathpreambles[UniLyXEntry(line).mathpreamble] = True
    for key in mathpreambles.keys():
        print key

def add_mathmacros(features=lyx20features):
    """
    Complete the unicodesymbols file with math macros
    """
    lastcodepoint = 159 # skip ASCII and adjacent non-printable chars
    for line in infile:
        lyxentry = UniLyXEntry(line)
        
        # keep lines without definition and lines aready defining mathmacro:
        if not(lyxentry.codepoint):
            outfile.write(line)
            continue
        # ensure monotonically ordered codepoints
        assert lyxentry.codepoint >= lastcodepoint, \
            'out of order at line\n%s' % line
        lastcodepoint = lyxentry.codepoint
        if lyxentry.mathmacro:
            outfile.write(line)
            continue

        # additional unimathsymbols.txt entries:
            
        for cp in range(lastcodepoint+1, lyxentry.codepoint):
            newentry = supported_mathmacro(cp, features)
            if newentry:
                outfile.write('%s\n' % newentry)
        lastcodepoint = lyxentry.codepoint + 1
    
        # current unimathsymbols.txt entry
        newentry = supported_mathmacro(lyxentry.codepoint, features)
        if newentry:
            lyxentry.flags = ','.join([lyxentry.flags, newentry.flags]
                                     ).strip(',')
            lyxentry.mathmacro = newentry.mathmacro
            lyxentry.mathpreamble = newentry.mathpreamble
            outfile.write('%s\n' % lyxentry)
            continue
        
        outfile.write(line)

    # "trailing" additional unimathsymbols.txt entries:
    for cp in unidata.sortedkeys():
        if cp <= lastcodepoint:
            continue
        newentry = supported_mathmacro(cp, features)
        if newentry:
            outfile.write('%s\n' % newentry)


# Default action / Tests
# ======================
# ::

if __name__ == '__main__':
    
    # report_differences()
    # reformat_unicodesymbols()
    list_mathpreambles()
    # add_mathmacros()

