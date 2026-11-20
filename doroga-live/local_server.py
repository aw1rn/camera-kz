#!/usr/bin/env python3
"""Optional localhost-only HEVC-to-H.264 adapter, using Python and FFmpeg."""
import http.server, urllib.request, urllib.parse, subprocess, tempfile, pathlib, threading, time, re, sys, shutil
ROOT=pathlib.Path(__file__).resolve().parent
IDS={'cam1','cam3','jjvezd','kunaev','semei','kostanay'}
cache={}; locks={key:threading.Lock() for key in IDS}
def converted(camera):
 with locks[camera]:
  previous=cache.get(camera)
  if previous and time.monotonic()-previous[0] < 12: return previous[1]
  url='https://kaztoll.kz/livestream/'+camera+'.mp4?t='+str(int(time.time()*1000))
  req=urllib.request.Request(url,headers={'User-Agent':'DorogaLive/1.0'})
  with urllib.request.urlopen(req,timeout=25) as src:
   video=src.read(50*1024*1024+1)
  if len(video)>50*1024*1024: raise ValueError('Видео слишком большое.')
  with tempfile.TemporaryDirectory(prefix='doroga-live-') as tmp:
   source=pathlib.Path(tmp)/'in.mp4'; dest=pathlib.Path(tmp)/'out.mp4';source.write_bytes(video)
   subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-nostdin','-y','-i',str(source),'-vf','scale=1280:-2','-c:v','libx264','-preset','ultrafast','-crf','23','-an','-movflags','+faststart',str(dest)],check=True,timeout=90,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
   result=dest.read_bytes()
  cache[camera]=(time.monotonic(),result)
  return result
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  path=urllib.parse.urlparse(self.path).path
  if path.startswith('/camera/') and path.endswith('.mp4'):
   cam=path[len('/camera/'):-len('.mp4')]
   if cam not in IDS: return self.send_error(404)
   try: data=converted(cam)
   except Exception as exc:
    print('Ошибка камеры:',str(exc),file=sys.stderr);return self.send_error(502,'Cannot obtain or convert camera video')
   start,end=0,len(data)-1;status=200
   if self.headers.get('Range'):
    m=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
    if not m:return self.send_error(416)
    start=int(m.group(1));end=min(int(m.group(2)) if m.group(2) else end,end)
    if start>end:return self.send_error(416)
    status=206
   self.send_response(status);self.send_header('Content-Type','video/mp4');self.send_header('Cache-Control','no-store');self.send_header('Accept-Ranges','bytes');self.send_header('Content-Length',str(end-start+1))
   if status==206:self.send_header('Content-Range',f'bytes {start}-{end}/{len(data)}')
   self.end_headers()
   try:self.wfile.write(data[start:end+1])
   except (BrokenPipeError,ConnectionResetError):pass
   return
  allowed={'/':'index.html','/index.html':'index.html','/style.css':'style.css','/app.js':'app.js'}
  if path not in allowed:return self.send_error(404)
  data=(ROOT/allowed[path]).read_bytes();mime={'html':'text/html; charset=utf-8','css':'text/css; charset=utf-8','js':'text/javascript; charset=utf-8'}[allowed[path].split('.')[-1]]
  self.send_response(200);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(data)
if __name__=='__main__':
 if not shutil.which('ffmpeg'):sys.exit('Установите FFmpeg и убедитесь, что ffmpeg доступен в PATH.')
 print('Откройте http://127.0.0.1:8765/?local=1  (для остановки нажмите Ctrl+C)',flush=True)
 http.server.ThreadingHTTPServer(('127.0.0.1',8765),Handler).serve_forever()
