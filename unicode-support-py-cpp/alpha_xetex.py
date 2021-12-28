#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
#             Released  without warranties or conditions of any kind
#             under the terms of the Apache License, Version 2.0
#             http://www.apache.org/licenses/LICENSE-2.0
# :Id: $Id:  $
# ::

"""Mathematical Alphanumeric Symbols in Unicode and LaTeX"""

import unicodedata
from symbols_xetex import *

# Configuration
# -------------

# I/O
outfile = file('../unimathalpha.tex', 'w')
# outfile = sys.stdout

# Include entries for characters without (listed) LaTeX macro? 
# ::

include_unsupported_chars = True
# include_unsupported_chars = False


# Include macros from the following packages::

packages = ['',
            'literal',
            # 'amsfonts', 
            # 'amssymb', 
            # 'amsmath',
            # 'amsxtra',
            'eufrak',
            # 'fourier',
            'isomath',
            # 'kpfonts',
            # 'kmath',
            # 'lxfonts',
            # 'mathabx',
            # 'mathcomp',
            # 'mathdesign',
            # 'MnSymbol'
            'bbold',
            'omlmathbf',
            'omlmathit',
            'omlmathrm',
            'omlmathsfit',
            'mathsfbf',
            'pzccal',
            # 'txfonts',
            # 'wrisym',
           ]


# Implementation
# --------------

outfile.write(preamble % (package_calls(packages), __doc__))
outfile.write("""
The table below lists Unicode symbols with category `mathalpha` and
arabic digits.
""")
outfile.write(used_features % ', '.join([pkg for pkg in packages 
                                         if pkg and pkg in features]))
outfile.write(used_packages % ', '.join([pkg for pkg in packages 
                                         if pkg not in features]))
outfile.write(table_head)

# table lines
for (key, entry) in parse_unimathsymbols.read_data():
    # skip non-alphanumeric symbols:
    if not entry.math_class or entry.math_class not in ('AN'):
        continue
    # skip ordinary symbols that are no digits:
    if (entry.math_class == 'N' and
        unicodedata.name(unichr(entry.codepoint), '').find('DIGIT') == -1):
        continue        
    cmd = entry.supported_cmd(packages)
    if not (cmd or include_unsupported_chars):
        continue     # skip unsupported chars
    if not cmd and entry.cmd:
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

if outfile != sys.stdout:
    print "Output written to", outfile.name
