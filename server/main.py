from http.server import HTTPServer, SimpleHTTPRequestHandler
 
def run(server_class=HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('', 8080)
    httpd = server_class(server_address, handler_class)
    print('HTTP server running on port 8080')
    httpd.serve_forever()
 
if __name__ == '__main__':
    run()