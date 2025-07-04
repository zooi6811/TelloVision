import cv2
import numpy as np
import time
import math

class LivePlot:
    def __init__(self, w=640, h=480, yLimit=[0,100],
                 interval=0.001, invert=False, char=' '):
        self.yLimit = yLimit
        self.w = w
        self.h = h
        self.invert = invert
        self.interval = interval
        self.char = char[0]
        self.imgPlot = np.zeros((self.h, self.w, 3), np.uint8)
        self.imgPlot[:] = 225,225,225
        cv2.rectangle(self.imgPlot, (0,0), (self.w,self.h), (0,0,0), cv2.FILLED)
        self.xP = self.yP = 0
        self.yList = []
        self.xList = list(range(100))
        self.ptime = 0

    def drawBackground(self):
        cv2.rectangle(self.imgPlot, (0,0), (self.w,self.h), (0,0,0), cv2.FILLED)
        cv2.line(self.imgPlot, (0,self.h//2), (self.w,self.h//2), (150,150,150),2)
        for x in range(0, self.w, 50):
            cv2.line(self.imgPlot, (x,0), (x,self.h), (50,50,50),1)
        for y in range(0, self.h, 50):
            cv2.line(self.imgPlot, (0,y), (self.w,y), (50,50,50),1)
            cv2.putText(self.imgPlot,
                        f'{int((self.h-y)*(self.yLimit[1]/self.h))}',
                        (10,y), cv2.FONT_HERSHEY_PLAIN, 1, (150,150,150),1)
        cv2.putText(self.imgPlot, self.char, (self.w-100, self.h-25),
                    cv2.FONT_HERSHEY_PLAIN, 5, (150,150,150),5)

    def update(self, y):
        if time.time() - self.ptime > self.interval:
            self.imgPlot[:] = 225,225,225
            self.drawBackground()
            cv2.putText(self.imgPlot, str(y), (self.w-125,50),
                        cv2.FONT_HERSHEY_PLAIN, 3, (150,150,150),3)
            if self.invert:
                self.yP = int(np.interp(y, self.yLimit, [self.h,0]))
            else:
                self.yP = int(np.interp(y, self.yLimit, [0,self.h]))
            self.yList.append(self.yP)
            if len(self.yList) == 100:
                self.yList.pop(0)
            for i in range(2, len(self.yList)):
                p1 = ((i-1)*(self.w//100)-(self.w//10), self.yList[i-1])
                p2 = ( i   *(self.w//100)-(self.w//10), self.yList[i])
                cv2.line(self.imgPlot, p1, p2, (255,0,255),2)
            self.ptime = time.time()
        return self.imgPlot
