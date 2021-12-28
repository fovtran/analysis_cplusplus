#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
#             Released  without warranties or conditions of any kind
#             under the terms of the Apache License, Version 2.0
#             http://www.apache.org/licenses/LICENSE-2.0
# :Id: $Id:  $
#
# ::

"""LaTeX math mode commands and corresponding Unicode characters"""

from symbols_xetex import *

# Configuration
# -------------

# I/O
outfile = file('../nameclashes.tex', 'w')
# outfile = sys.stdout

# Include entries for characters without (listed) LaTeX macro?
# ::

include_unsupported_chars = True
# include_unsupported_chars = False

# Include macros from the following packages::

packages = ['',
            'amssymb',
            'amsmath',
            # 'amsxtra',
            # 'bbold',
            # 'esint',
            # 'fourier',
            # 'isomath',
            # 'lxfonts',
            'mathabx',
            # 'mathcomp',
            # 'mathdesign',
            # 'mathdots',
            # 'MnSymbol',
            # 'oz',
            'stmaryrd',
            'wasysym',
            # 'wrisym',
           ]


# Implementation
# ---------------

intro = r"""
\subsection*{name clashes}

Commands that produce different symbols with different math packages.
Depending on the package implementation, either the last package "wins"
or throws an error.
"""

table_head = r"""
\begin{longtable}{llcclll}
\toprule
Command & No. & Text & Math & Category & Requirements & Comments\\
\midrule
\endhead
"""

# Parse and sort data::

data = parse_unimathsymbols.read_data()
cmds = parse_unimathsymbols.sort_by_command(data)

# sort into lists
cmdlists = parse_unimathsymbols.Table()
for (key, entry) in cmds:
    # skip mappings of plain input chars to ``\mathnormal``:
    if entry.requirements.count('-literal'):
        continue
    try:
        cmdlists[entry.cmd].append(entry)
    except KeyError:
        cmdlists[entry.cmd] = [entry]

# add clashing unicode-math cmds::

for entry in data.values():
    if (entry.unicode_math                      # unicode-math cmd exists
        and entry.unicode_math != entry.cmd     # differs from standard cmd
        and cmdlists.get(entry.unicode_math)): # which exists as well
        
        # no clash, if both cmds point ot the same symbol
        if entry.codepoint in [entry.codepoint 
                               for entry in cmdlists[entry.unicode_math]]:
            continue
        # Add a new entry for the clashing unicode-math command
        newentry = parse_unimathsymbols.new_entry(entry.codepoint)
        newentry.cmd = entry.unicode_math
        newentry.requirements = 'unicode-math'
        try:
            cmdlists[newentry.cmd].append(newentry)
        except KeyError:
            cmdlists[newentry.cmd] = [newentry]


# find doublettes:
clashes = [key for key in cmdlists.sortedkeys()
           if len(cmdlists[key]) > 1]

# preamble + table header::

outfile.write(preamble % (package_calls(packages), __doc__))
outfile.write(intro)
outfile.write(used_packages % ', '.join([pkg for pkg in packages if pkg]))
outfile.write(table_head)



# table lines::
for cmd in clashes:
    for entry in cmdlists[cmd]:
        macro = macro_example(entry, packages)
        line = ' & '.join([txt2tex(entry.cmd),
                        '%05X' % entry.codepoint,
                        txt2tex(entry.utf8),
                        macro,
                        entry.category,
                        entry.requirements,
                        txt2tex(entry.comment),
                        ]) + r' \\'
        outfile.write(line + '\n')
    
# the end
outfile.write(table_foot)
outfile.write(r'\end{document}')

if outfile != sys.stdout:
    print "%d commands, %d clashes" % (len(cmds), len(clashes))
    print "Output written to ", outfile.name
