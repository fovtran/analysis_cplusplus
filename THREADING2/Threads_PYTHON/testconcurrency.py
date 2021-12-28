import sys
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import time

import concurrent.futures
import urllib.request

URLS = ['http://www.tmall.com/',
        'http://www.cnn.com/',
        'http://huawei.com/',
        'http://www.bbc.co.uk/',
        'http://jd.com/',
        'http://weibo.com/?c=spr_web_360_hao360_weibo_t001',
        'http://www.sina.com.cn/',
        'http://taobao.com',
        'http://www.amazon.cn/?tag=360daohang-23',
        'http://www.baidu.com/',
        'http://www.pconline.com.cn/?ad=6347&360hot_site']

# Retrieve a single page and report the url and contents
def load_url(url, timeout):
    conn = urllib.request.urlopen(url, timeout=timeout)
    return conn.readall()




class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        QTimer.singleShot(3000, self.speedCalculate)

#        self.timerC = QTimer();
#        self.timerC.timeout.connect(self.speedCalculate)
#        self.timerC.start(1000)




    def speedCalculate(self):#compare with for loop
        t1=time.clock()
        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(URLS)) as executor:
            # Start the load operations and mark each future with its URL
            future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (url, exc))
                else:
                    print('%r page is %d bytes' % (url, len(data)))

        t2=time.clock()
        print('t2-t1-------------', t2-t1)
#
#        for url in URLS:
#            data=load_url(url, 60)
#            print(url, len(data))
#
#        t3=time.clock()
#        print('t3-t2-------------', t3-t2)
    #

	def speedCalculate(self):
		threading.Thread(target=self._speedCalculate).start()

	def _speedCalculate(self):#compare with for loop
		t1=time.clock()
		# We can use a with statement to ensure threads are cleaned up promptly
		with concurrent.futures.ThreadPoolExecutor(max_workers=len(URLS)) as executor:
			# Start the load operations and mark each future with its URL
			future_to_url = {executor.submit(load_url, url, 60): url for url in URLS}
			for future in concurrent.futures.as_completed(future_to_url):
				url = future_to_url[future]
				try:
					data = future.result()
				except Exception as exc:
					print('%r generated an exception: %s' % (url, exc))
				else:
					print('%r page is %d bytes' % (url, len(data)))

		t2=time.clock()
		print('t2-t1-------------', t2-t1)

if __name__ == '__main__':
    app =QApplication(sys.argv)
    splitter =MainWindow()
    splitter.show()
    app.exec_()