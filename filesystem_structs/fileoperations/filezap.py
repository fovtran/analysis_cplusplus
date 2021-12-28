#!python
#
import os, sys, stat
import optparse
import logging

usage = '''%s [ -z | --zapdestroyall ] path''' % sys.argv[0]

def oswalk(path):
	'''dir walk
	   linux callback(arg, directory, files)
	   windows (directory, dirnames, files) = os.walk(path)'''
	   
	for directory, dirnames, filenames in os.walk(path):
		logging.warning('processing %s' % directory)
		for f in filenames:
			logging.info('retrieving %s' % directory)
			dest = os.path.join(directory, f)
			zero(dest, '') # '\x00' or '\xff'
	
def zero(path, format):
	'''Destroys all file content and zaps file data to new'''
	logging.warning('destroying all data at %s' % path)
	try:
		st = os.lstat(path)
	except os.error:
		pass
	if stat.S_ISREG(st.st_mode):
		(mode, ino, dev, nlink, uid, gid, size, atime, mtime, ctime) = os.stat(path)
		todo = size
		logging.info('zeroing %s' %path)
		f = open(path, 'w')
		f.write(format)
		f.close()

if __name__ == '__main__':
	parser = optparse.OptionParser(usage=usage, version='0.1c')
	parser.add_option('-z', '--zero', dest='path', default='/tmp', type='string', help='zaps and destroys data in a file structure')
	(options, args) = parser.parse_args()
	
	if len(options.path)>0:
		oswalk(options.path)
		
