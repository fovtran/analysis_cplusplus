#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
#             Released  without warranties or conditions of any kind
#             under the terms of the Apache License, Version 2.0
#             http://www.apache.org/licenses/LICENSE-2.0
# :Id: $Id:  $

import sys, re
import parse_unimathsymbols

# Configuration
# -------------

# I/O
outfile = file('../unicode-math-diff.rst', 'w')
# outfile = sys.stdout


# Implementation
# ---------------

# preamble + header

header = [line.replace('# ', '').strip('#')
          for line in parse_unimathsymbols.read_header()]

table_header = """
.. csv-table::
  :delim: ^
  :header: %s
""" % header[-1].replace('^', ',')

def write_section(header, data):
    """write a section to the rst outfile

    `header` is the leading string (section title, adornment and intro)
    `data` is a Table instance written to a csv_table
    """
    outfile.write(header + table_header)
    for key, entry in data:
        packages = ', '.join(entry.provided_by() + 
                             ['-' + pkg for pkg in entry.conflicts_with()])
        if entry.cmd and not packages:
            packages = '\*'
        row = [r'%05X' % entry.codepoint,
               r'\%s'  % entry.utf8,
               r'\%s'  % entry.cmd,
               r'\%s'  % entry.unicode_math,
               r'%s'   % entry.math_class,
               r'%s'   % entry.category,
               r'%s'   % packages,
               r'\%s'  % entry.comment.replace('\\', '\\\\'),
              ]
        outfile.write('  %s\n' % '^'.join(row))

# Get data
data = parse_unimathsymbols.read_data()

# use substitution for delimiting character ('^')
data[ord('^')].utf8 = ' |circum|' # leading space prevents escaping by '\'

# "Bins":
new_cmds = parse_unimathsymbols.Table()
missing_cmds = parse_unimathsymbols.Table()
new_aliases = parse_unimathsymbols.Table()
other_names = parse_unimathsymbols.Table()
mathalphabets = parse_unimathsymbols.Table()
mathalphabet_cmds = parse_unimathsymbols.Table()

for (key, entry) in data:
    if re.match(r'\\math[a-z]+\{.*\}', entry.cmd):
        mathalphabet_cmds[key] = entry
        mathalphabets[entry.cmd[:entry.cmd.find('{')]] = entry
    elif not entry.unicode_math:
        if entry.cmd == entry.utf8:
            continue
        missing_cmds[key] = entry
    elif not entry.cmd:
        new_cmds[key] = entry
    elif entry.cmd != entry.unicode_math:
        if entry.unicode_math in [e.cmd for e in entry.related_commands('=')
                                 if e.requirements == '']:
            continue # unicode-math uses a standard alias
        if (entry.cmd == entry.utf8) or (entry.cmd[1:] == entry.utf8):
            new_aliases[key] = entry
        else:
            other_names[key] = entry


outfile.write("""\
Differences to unicode-math
===========================


Summary
-------

The unicode-math_ package for XeTeX and LuaTeX supports OpenType
Unicode fonts also for math typesetting. With this package, literal
Unicode symbols can be used in math mode.

.. _unicode-math:
   http://mirror.ctan.org/help/Catalogue/entries/unicode-math.html

It also defines %d commands for symbols with math usage. 
Most of the commands correspond to the ones in «traditional» (La)TeX
and established math packages like amssymb. The others are

| %d new commands for symbols not supported by traditional LaTeX,
| %d `new aliases`_ for ASCII characters,
| %d commands that use `command names differing from traditional LaTeX`_,
| %d commands for symbols that can be represented via %d `math alphabets`_.

The database also lists %d math-related `symbols missing in unimath-symbols`_.

.. |circum| unicode:: 0x5E .. circumflex character (delimiter)
""" % (len(data) - len(missing_cmds),
       len(new_cmds),
       len(new_aliases),
       len(other_names),
       len(mathalphabet_cmds),
       len(mathalphabets),
       len(missing_cmds),
      ))

write_section("""
new aliases
-----------
""", new_aliases)

write_section("""
command names differing from traditional LaTeX
----------------------------------------------
""", other_names)

write_section("""
symbols missing in unimath-symbols
----------------------------------

The following %d symbols are mentioned in MathClassEx_ or
available via math mode commands but not listed in unimath-symbols_:

.. _MathClassEx:
   http://www.unicode.org/Public/math/revision-11/MathClassEx-11.txt
.. _unimath-symbols:
   http://mirror.ctan.org/macros/latex/contrib/unicode-math/unimath-symbols.pdf
""" % len(missing_cmds), missing_cmds)


write_section("""
math alphabets
--------------

The table lists example entries for each math alphabet that can
be realized with «traditional» LaTeX.
%d `alphanumerical math symbols`_ can be represented via math %d alphabets.

.. _alphanumerical math symbols: unimathalpha.pdf
""" % (len(mathalphabet_cmds), len(mathalphabets)), mathalphabets)


if outfile != sys.stdout:
    print "Output written to", outfile.name
