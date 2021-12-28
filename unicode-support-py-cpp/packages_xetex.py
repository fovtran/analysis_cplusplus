#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
#             Released  without warranties or conditions of any kind
#             under the terms of the Apache License, Version 2.0
#             http://www.apache.org/licenses/LICENSE-2.0
# :Id: $Id:  $

"""Math symbols defined by LaTeX package «%s»"""

from symbols_xetex import *

# Configuration
# -------------

# I/O
logfile = open('../math-packages.txt', 'w')
outdir = 'mathpackages/'
texfile_pattern = '%s-symbols.tex'
pdffile_pattern = '%s-symbols.pdf'

# Include entries for characters without (listed) LaTeX macro? ::

# include_unsupported_chars = True
include_unsupported_chars = False

# For debugging/development: update just one package file::

only_package = ''
# only_package = 'arevmath'

comments = {
'arevmath': 'Sans serif math font based on Vera Sans.',
'fourier': r"""Capital Greek letters do not change shape in math alphabets. \\
\verb+\pounds+ prints dollar sign (\$).
""",
'kmath': 'Part of the «Kerkis» font package. Requires txfonts.',
'kpfonts': 'List not complete (check kpfonts.pdf page 8).',
'mathpazo': 'Capital Greek letters cannot be used in math alphabets.',
'oz': r"""Clash with package xunicode:
Defines \verb+\TH+, the established LICR for Þ (capital thorn),
as \verb+\boldword{theorem}+.""",
'wrisym': 'does not work with XeTeX',
}
comments['fouriernc'] = comments['fourier']
comments['mathptmx'] = comments['mathpazo']

# Some packages provide alternative glyphs for the standard symbols.
# These should be shown in the survey::

math_font_packages = ['MnSymbol',
                      'arevmath',
                      # 'esint',
                      'fourier',
                      'fouriernc',
                      'kmath',
                      'kpfonts',
                      'mathdesign',
                      # 'mathdots',
                      'mathpazo',
                      'mathptmx',
                      'pxfonts',
                      'qpxmath',
                      'qtxmath',
                      'tmmath',
                      'txfonts']

# Implementation
# ---------------

data = parse_unimathsymbols.read_data()
cmds = parse_unimathsymbols.sort_by_command(data)

# Get list of all mentioned packages::

pkg_table = parse_unimathsymbols.Table({'': 0})

for entry in cmds.values():
    if not entry.requirements:
        pkg_table[''] += 1 # standard commands
    for pkg in entry.provided_by():
        if pkg in features:
            continue
        try:
            pkg_table[pkg] += 1
        except KeyError:
            pkg_table[pkg] = 1

# for debugging/development: update just one package file
if only_package:
    pkg_table = parse_unimathsymbols.Table({only_package: True})

# Write a tex file for every listed math package::

for pkg in pkg_table.sortedkeys():
    packages = [pkg]
    extrapkgs = []
    # special cases:
    # packages to skip:
    if pkg in ('omlmathsfbf', # ∄ OML encoded sans bold upright font
               'tmmath',      # commercial fonts
               # 'wrisym',    # clash with XeTeX, skiped after logfile update
              ):
        continue
    # let \mathbb work for small greek letters
    if pkg in ('bbold', 'mathbbol'):
        extrapkgs = ['fixmath']
    # show also package-provided glyphs for standard commands
    if pkg in math_font_packages:
        packages.append('')

    # update logfile
    pdfname = pdffile_pattern % (pkg or 'standard')
    logfile.write('* `%s <%s>`_ (%d)' %
                  (pdfname, outdir + pdfname, pkg_table[pkg]))
    if pkg in math_font_packages:
        logfile.write(' [#alternative-glyphs]_')
    logfile.write('\n')

    if pkg == 'wrisym':      # no PDF with XeTeX
        continue

    # generate latex source file:
    outfile = file('../' + outdir + texfile_pattern % (pkg or 'standard'), 'w')
    outfile.write(preamble % (package_calls(packages+extrapkgs),
                              __doc__ % pkg))
    outfile.write(comments.get(pkg, ''))
    outfile.write(table_head)

    # table lines
    for (key, entry) in parse_unimathsymbols.read_data():
        cmd = entry.supported_cmd(packages)
        if not cmd:
            cmd = entry.substitution_cmd(packages)
            if cmd:
                cmd = '(%s)' % cmd
            elif not include_unsupported_chars:
                continue     # skip unsupported chars
        if cmd:
            # add a base character to accent and radical macros:
            if entry.category in ('mathaccent', 'mathradical',
                                'mathover', 'mathunder'):
                cmd += '{x}'
                cmd = cmd.replace('){x}', '{x})')
            if entry.category in ('mathopen'):
                cmd = cmd.replace(r'\lgroup', r'\left\lgroup\right.')
            if entry.category in ('mathclose'):
                cmd = cmd.replace(r'\rgroup', r'\left.\right\rgroup')
        elif entry.cmd:
            cmd = '[na]' # not available with current package selection
        line = ' & '.join(['%05X' % entry.codepoint,
                        txt2tex(entry.utf8),
                        '$%s$' % cmd,
                        txt2tex(entry.cmd),
                        entry.category,
                        entry.requirements,
                        txt2tex(entry.comment),
                        ]) + r' \\'
        outfile.write(line + '\n')

    # the end
    outfile.write(table_foot)
    outfile.write(r'\end{document}')
    outfile.close()
