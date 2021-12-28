import concurrent.futures
import urllib.request

URLS = ['http://www.tmall.com/',
        'http://www.pconline.com.cn/?ad=6347&360hot_site']

#QTimer.singleShot(3000, self.speedCalculate)
#self.timerC = QTimer();
#self.timerC.timeout.connect(self.speedCalculate)
#self.timerC.start(1000)

def decode_image(frame):
	myframe = np.fromstring(frame, dtype=np.uint8)
	outputstream = self.OpenCVTools.imdecode(myframe)
	return outputstream

    def future_decode_image(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(URLS)) as executor:
            future_to_image = {executor.submit(decode_image, f): f for frame in frames}
            for future in concurrent.futures.as_completed(future_to_image):
                self.outputstream = future_to_image[future]
                try:
                    data = future.result()
                except (Exception as exc):
                    print('image decoder generated an exception: %s' % (exc))
                else:
                    print(len(data))