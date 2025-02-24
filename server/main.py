from http.server import HTTPServer, SimpleHTTPRequestHandler
import cgi
import shutil
import os

class ChessRequestHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST'})
        self.send_response(200)
        self.send_header('Content-type', 'text/html')

        self.end_headers()
        #shutil.copyfile(self.wfile)
        self.wfile.write(b'File uploaded successfully')
    def do_GET(self):
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'GET'})
        self.send_response(200)
        self.send_header('Content-type', 'text/html')

        self.end_headers()
        fd=os.open("./htmls/index.html",os.O_RDONLY)
        self.wfile.write(os.read(fd,fd.bit_length()))
        os.close(fd)

def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('', 8080)
    httpd = server_class(server_address, handler_class)
    print('HTTP server running on port 8080')
    httpd.serve_forever()
 
if __name__ == '__main__':
    run(handler_class=ChessRequestHandler)