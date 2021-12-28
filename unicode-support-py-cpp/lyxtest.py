#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
# :Licence:   This work may be distributed and/or modified under the
#             conditions of the `LaTeX Project Public License`_,
#             either version 1.3 of this license or (at your option)
#             any later version.
# ::

"""Test LyX's `unicodesymbols' mappings"""

import os, sys
import parse_unimathsymbols
from parse_lyx_unicodesymbols import UniLyXEntry
from symbols_xetex import *

# Configuration
# -------------

# I/O
lyxunifile = file(os.path.expanduser('~/.lyx-svn/unicodesymbols'), 'r')
outfile = file(os.path.expanduser('~/Texte/Test/LyX/unimathtest.tex'), 'w')
# outfile = sys.stdout

# Include macros from the following packages::

packages = ['',
            'amssymb', 
            'amsmath',
            # 'amsxtra',
            # 'bbold',
            'esint',
            # 'fourier',
            # 'gensymb',
            # 'isomath',
            # 'kmath',
            # 'kpfonts',
            # 'lxfonts',
            # 'mathabx',
            # 'mathcomp',
            # 'mathdesign',
            # 'mathdots',
            # 'MnSymbol',
            # 'oz',
            # 'txfonts',
            # 'stmaryrd',
            'wasysym',
            # 'wrisym',
           ]


# Implementation
# ---------------

preamble = r"""\documentclass[a4paper]{article}
\usepackage{fixltx2e}

%% Requirements
%s

%% %% Text font
%% \usepackage[no-math]{fontspec}
%% \usepackage{xunicode}
%% \setmainfont[BoldFont={XITS Bold},ItalicFont={XITS Italic}]{XITS Math}
%% \setsansfont{DejaVu Sans}
%% %% \setmonofont[HyphenChar=None,Scale=MatchUppercase]{DejaVu Sans Mono}
%% \setmonofont[HyphenChar=None,Scale=MatchUppercase]{FreeMono}

\begin{document}

\section*{%s}
"""


used_packages = r"""
Used packages: \texttt{%s}.
"""
outfile.write(preamble % (package_calls(packages) +
                          '\n\\usepackage[utf8]{inputenc}', # to import as UTF8
                          __doc__))
if [pkg for pkg in packages if pkg and pkg in features]:
    outfile.write(used_features % ', '.join([pkg for pkg in packages 
                                             if pkg and pkg in features]))
outfile.write(used_packages % ', '.join([pkg for pkg in packages 
                                         if pkg not in features]))

outfile.write(r"""

\begin{description}
\item[codepoint] \textbf{Text, Math, Name/Comments, (mathpreamble)}
""")

# lines
# -----------

for line in lyxunifile:
    lyxentry = UniLyXEntry(line)
    # skip, if there is no codepoint
    if not lyxentry.codepoint:
        continue
    # skip, if there is no math part
    # if not lyxentry.mathmacro:
    #     continue
    if lyxentry.mathpreamble is None:
        lyxentry.mathpreamble = ''
    if lyxentry.comment is None:
        lyxentry.comment = ''
    # skip, if required packages are not supported
    preamble = lyxentry.mathpreamble.replace('esintoramsmath', 'esint amsmath')
    preamble += ',' + lyxentry.textpreamble
    # combinations: ',' -> AND, ' ' -> OR
    for requirement in preamble.split(','):
        unsupported = [True for pkg in requirement.strip().split()
                       if pkg not in packages]
        if unsupported:
            print '%s not supported: %s' % (requirement, lyxentry)
            break
    if unsupported:
        continue
    
    utf8 = unichr(lyxentry.codepoint).encode('utf8')
    line = '\\item[%s] %s, %s: %s, %s\n' % ('%05X' % lyxentry.codepoint,
                                            txt2tex(utf8),
                                            '$%s$' % utf8,
                                            txt2tex(lyxentry.comment),
                                            txt2tex(lyxentry.mathpreamble),
                                           )
    outfile.write(line)

# the end
outfile.write(r"""
\end{description}
\end{document}
""")

if outfile != sys.stdout:
    print "Output written to ", outfile.name
