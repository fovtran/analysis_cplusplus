#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2011 Günter Milde.
#             Released  without warranties or conditions of any kind
#             under the terms of the Apache License, Version 2.0
#             http://www.apache.org/licenses/LICENSE-2.0

# Write Python source for LaTeX->Unicode translation tables
# =========================================================
#
# Generate dictionaries for replacing LaTeX math commands with the
# corresponding Unicode character.

import sys, unicodedata
import parse_unimathsymbols

# Configuration
# -------------

# Include macros from the following packages::

packages = ['',
            # 'literal', # "feature", no real packge
            'amssymb',
            'amsmath',
            'amsxtra',
            'bbold',
            'esint',
            # 'fourier',
            # 'gensymb',
            # 'isomath',
            # 'kmath',
            # 'lxfonts',
            'mathabx',
            # 'mathcomp',
            # 'mathdesign'
            'mathdots',
            # 'MnSymbol'
            # 'omlmathit',
            # 'pzccal',
            # 'txfonts',
            'stmaryrd',
            'wasysym',
            # 'wrisym',
           ]

# reverse the package list, so that the standard and ams* package commands
# "win" in case of name clashes:
packages.reverse()

# I/O
outfile = file('tex2unichar.py', 'w')
# outfile = sys.stdout

# append package name to outfile, if there is only one package
# if len(packages) == 1 and outfile.endswith('.py'):
#     outfile = outfile.replace('.py', packages[0] + '.py')

# Implementation
# ---------------

# preamble + header

header = """\
# -*- coding: utf8 -*-

# LaTeX math to Unicode symbols translation dictionaries.
# Generated with ``write_tex2unichar.py`` from the data in
# http://milde.users.sourceforge.net/LUCR/Math/

# Includes commands from: %s

""" % ', '.join([pkg or 'standard LaTeX' for pkg in packages])

outfile.write(header)

def fill_tables(data, tables):
    """Sort `data` entries into `tables` according to math category."""
    for (key, entry) in data:
        if not entry.is_supported(packages):
            continue     # skip unsupported chars
        cmd = entry.cmd
        # sort only "real" commands, without arguments:
        if not cmd.startswith('\\'):
            continue
        if '{' in cmd and '}' in cmd:
            continue # TODO: dictionaries for math alphabets
        # if '[' in cmd and ']' in cmd:  # optional arguments
        #     continue        
        # Spaces (have no math category):
        if not entry.category and entry.math_class == 'S':
            entry.category = 'space'
        # create table for math category:
        if entry.category not in tables:
            tables[entry.category] = parse_unimathsymbols.Table()
        # only list first occurence, never overwrite (favours low code points)
        if cmd not in tables[entry.category]:
            tables[entry.category][cmd] = entry


# data tables

data = parse_unimathsymbols.read_data()
cmds = parse_unimathsymbols.sort_by_command(data)
ersatzcmds = parse_unimathsymbols.substitution_commands(data)

print ersatzcmds[r'\hbar']

tables = parse_unimathsymbols.Table()
fill_tables(cmds, tables)    
fill_tables(ersatzcmds, tables)    

for category, table in tables:
    outfile.write(category + ' = {\n')
    for cmd, entry in table:
        row = "    '%s': u'%s', # %s %s\n" % (
            cmd.lstrip('\\'), 
            unichr(entry.codepoint).encode('unicode_escape'), 
            entry.utf8,
            unicodedata.name(unichr(entry.codepoint)))
        outfile.write(row)
    outfile.write('    }\n')

if outfile != sys.stdout:
    print "Output written to", outfile.name
