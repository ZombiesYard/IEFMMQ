import socket
import time
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(b"pnt_331", ("127.0.0.1", 7778)) 
#wait 3 seconds

#time.sleep(3)
#sock.sendto(b"CLEAR",   ("127.0.0.1", 7778))   