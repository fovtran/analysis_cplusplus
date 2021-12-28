#!/usr/bin/env python
# -*- coding: utf8 -*-
# :Copyright: © 2008 Günter Milde.
#             Released  without warranties or conditions of any kind
#             under the terms of the Apache License, Version 2.0
#             http://www.apache.org/licenses/LICENSE-2.0
# :Id: $Id:  $

# ===========================================================================

# Parse the Unicode math classes file.

import sys, re, difflib, unicodedata
import parse_unimathsymbols

data = parse_unimathsymbols.read_data()

infile = file('../references/MathClassEx-14.txt', 'r')


for line in infile:
    # skip comment lines
    if line.startswith('#'):
        continue
    # parse lines into fields
    try:
        (No, 
         math_class, 
         utf8, 
         entity_name, 
         entity_set, 
         comments, 
         name) = [i.strip() for i in line.split(';')]
    except ValueError:
        if line.strip():
            print "error in line: '%s'" % line
            raise
        else:
            continue
    
    # expand ranges
    numbers = [int(n, 16) for n in No.split('..')]
    if len(numbers) == 2:
        numbers = range(numbers[0], numbers[1]+1)

    for number in numbers:
        try:
            entry = data[number]
        except KeyError:
            try:
                entry = parse_unimathsymbols.new_entry(number)
            except ValueError: # non existent Unicode char in range
                continue
            if entity_name:
                entry.comment = entity_name
            if comments and comments.find('compatibility variant') == -1:
                entry.comment += ' ' + comments
        
        entry.math_class = math_class
        # push back to data
        data[number] = entry


# Write back
# ----------

header = parse_unimathsymbols.read_header()

# Test for differences after a read-write cycle. Whitespace adjacent to the
# delimiter is not significant. ::

in_lines = file(parse_unimathsymbols.datafilename, 'r').readlines()
in_lines = [re.sub(r'[ \t]*\^[ \t]*', '^', line)
            for line in in_lines]

header = [re.sub(r' *\^ *', '^', line) for line in header]

# print "header", "".join(header)

out_lines = [str(v)+'\n' for (k,v) in data]

diff = ''.join(difflib.unified_diff(in_lines, header + out_lines,
                                    parse_unimathsymbols.datafilename, 
                                    '*round trip*'))
if diff:
    print diff
else:
    print 'no differences after round trip'

# Write back to outfile::

outfile = None
# outfile = sys.stdout
# outfile = file('../data/unimathsymbols.txt', 'w')
if outfile:
    data.header = header
    parse_unimathsymbols.write_data(data, outfile)
    if outfile != sys.stdout:
        print "Output written to", outfile.name


# for (key, entry) in sort_by_command(data):
#     print entry


