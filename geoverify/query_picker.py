"""First-frame point selection only; no robot/controller commands."""
import os
import sys
import numpy as np


class PointSelection:
    def __init__(self,width,height,display_width,display_height):
        if min(width,height,display_width,display_height)<1:
            raise ValueError('Positive image dimensions required')
        self.width,self.height=width,height
        self.display_width,self.display_height=display_width,display_height
        self.points=[]

    def add(self,x,y):
        if not 0<=x<self.display_width or not 0<=y<self.display_height:
            return
        # Invert OpenCV resize's pixel-center coordinates, not edge coordinates.
        px=np.clip((x+.5)*self.width/self.display_width-.5,0,self.width-1)
        py=np.clip((y+.5)*self.height/self.display_height-.5,0,self.height-1)
        self.points.append((float(px),float(py)))

    def undo(self):
        if self.points:
            self.points.pop()

    def queries(self):
        if not self.points:
            raise ValueError('Select at least one query point')
        return np.c_[np.zeros(len(self.points)),np.asarray(self.points)]


def pick_queries(rgb,max_width=1280,max_height=800):
    rgb=np.asarray(rgb)
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8:
        raise ValueError('RGB uint8 first frame required')
    if sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
        raise RuntimeError('Interactive selection needs a graphical DISPLAY; use --queries or --mask on a headless server')
    import cv2
    h,w=rgb.shape[:2]
    scale=min(1.,max_width/w,max_height/h)
    dw,dh=max(1,round(w*scale)),max(1,round(h*scale))
    selection=PointSelection(w,h,dw,dh)
    title='GeoVerify: left add | right undo | Enter confirm | Esc cancel'
    base=cv2.resize(rgb[:,:,::-1],(dw,dh))
    def callback(event,x,y,flags,param):
        if event==cv2.EVENT_LBUTTONDOWN:
            selection.add(x,y)
        elif event==cv2.EVENT_RBUTTONDOWN:
            selection.undo()
    cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)
    try:
        cv2.setMouseCallback(title,callback)
        while True:
            frame=base.copy()
            for i,(x,y) in enumerate(selection.points):
                px=round((x+.5)*dw/w-.5)
                py=round((y+.5)*dh/h-.5)
                cv2.circle(frame,(px,py),4,(0,255,255),-1)
                cv2.putText(frame,str(i),(px+5,py),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,255,255),1)
            cv2.imshow(title,frame)
            key=cv2.waitKey(20)&255
            if key in (10,13) and selection.points:
                return selection.queries(),None  # No inferred object memberships.
            if key==27 or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                raise RuntimeError('Query selection cancelled')
            if key in (8,127):
                selection.undo()
    finally:
        cv2.destroyWindow(title)
