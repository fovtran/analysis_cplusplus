import gc, traceback, sys, code, signal, faulthandler, pdb, logging,  unittest, atexit

logger = logging.getLogger(__name__)

def cleanup():
	logger.log(msg="Running cleanup...", level=3)

atexit.register(cleanup)

pdb.set_trace()
#pdb.post_mortem()
#pdb.pm()
#assert list_id+3 == id(item), "Oh no! This assertion failed!"
#traceback.print_exc(file=sys.stdout)

gc.set_debug(gc.DEBUG_SAVEALL)

print(gc.get_count())
lst = [1,2,3,4,5,6,7,8,9,0]
lst.append(lst)
list_id = id(lst)
del lst
gc.collect()

for item in gc.garbage:
	print(item)
	assert list_id == id(item), "Oh no! This assertion failed!"
	print([list_id == id(item)])

class TestStringMethods(unittest.TestCase):

  def test_upper(self):
      self.assertEqual('foo'.upper(), 'FOO')

  def test_isupper(self):
      self.assertTrue('FOO'.isupper())
      self.assertFalse('Foo'.isupper())

  def test_split(self):
      s = 'hello world'
      self.assertEqual(s.split(), ['hello', 'world'])
      # check that s.split fails when the separator is not a string
      with self.assertRaises(TypeError):
          s.split(2)

# unittest.main()

def debug(sig, frame):
  id2name = dict((th.ident, th.name) for th in threading.enumerate())
  #for threadId, stack in sys._current_frames().items():
    #print(id2name[threadId])
    #traceback.print_stack(f=stack)

signal.signal(signal.SIGUSR1, debug)  # Register handler
signal.signal(signal.SIGQUIT, debug)  # Register handler
#signal.signal(signal.SIGINT, debug)
signal.signal(signal.SIGINT, lambda sig, frame: pdb.Pdb().set_trace(frame))
faulthandler.register(signal.SIGUSR1)
faulthandler.register(signal.SIGQUIT)
faulthandler.register(signal.SIGINT)

try:
	assert list_id+3 == id(item), "Oh no! This assertion failed!"
except:
	#print ('print_exception():')
	exc_type, exc_value, exc_tb = sys.exc_info()
	#traceback.print_exception(exc_type, exc_value, exc_tb)
	print(exc_type, exc_value)
