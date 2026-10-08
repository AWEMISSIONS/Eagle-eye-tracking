from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import cv2

ZONE_COLORS = {
    "road": (255, 170, 0),
    "driveway": (0, 255, 255),
    "sidewalk": (255, 100, 220),
    "ignore": (120, 120, 120),
}

class ZoneManager:
    def __init__(self, path: Path):
        self.path = path
        self.zones = {"road": None, "driveway": None, "sidewalk": None, "ignore": None}
        self.load()

    def load(self):
        try:
            if self.path.exists():
                raw=json.loads(self.path.read_text(encoding='utf-8'))
                for k in self.zones:
                    v=raw.get(k)
                    if isinstance(v,list) and len(v)==4:
                        self.zones[k]=[max(0.0,min(1.0,float(x))) for x in v]
        except Exception:
            pass

    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.path.write_text(json.dumps(self.zones,indent=2),encoding='utf-8')

    def set_zone(self, name: str, rect_norm):
        if name not in self.zones:
            return
        x1,y1,x2,y2=rect_norm
        x1,x2=sorted((max(0,min(1,x1)),max(0,min(1,x2))))
        y1,y2=sorted((max(0,min(1,y1)),max(0,min(1,y2))))
        if x2-x1 < .01 or y2-y1 < .01:
            return
        self.zones[name]=[x1,y1,x2,y2]
        self.save()

    def clear(self, name: str):
        if name in self.zones:
            self.zones[name]=None
            self.save()

    def zone_at(self, center, shape) -> str:
        h,w=shape[:2]
        x=center[0]/max(w,1); y=center[1]/max(h,1)
        # ignore has highest priority, driveway next, then sidewalk/road
        for name in ("ignore","driveway","sidewalk","road"):
            r=self.zones.get(name)
            if r and r[0] <= x <= r[2] and r[1] <= y <= r[3]:
                return name
        return ""

    def draw(self, frame):
        h,w=frame.shape[:2]
        for name,r in self.zones.items():
            if not r:
                continue
            x1,y1,x2,y2=int(r[0]*w),int(r[1]*h),int(r[2]*w),int(r[3]*h)
            c=ZONE_COLORS[name]
            cv2.rectangle(frame,(x1,y1),(x2,y2),c,2)
            cv2.putText(frame,name.upper(),(x1+5,max(22,y1+22)),cv2.FONT_HERSHEY_SIMPLEX,.58,c,2)
