import http.server
import io
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
import numpy as np
import cgi
from run_image_predictor import apply_sam2_mask

PORT = 8002

class SimpleCORSRequestHandler(BaseHTTPRequestHandler):
    def _set_cors_headers(self):
        origin = self.headers.get('Origin')
        if origin:
            self.send_header('Access-Control-Allow-Origin', origin)
        self.send_header('Access-Control-Allow-Credentials', 'true')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With, Content-Type')
        
    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_POST(self):
        # Parse headers and ensure multipart/form-data
        ctype, pdict = cgi.parse_header(self.headers['Content-Type'])
        if ctype == 'multipart/form-data':
            pdict['boundary'] = bytes(pdict['boundary'], "utf-8")
            content_length = int(self.headers['Content-Length'])
            pdict['CONTENT-LENGTH'] = content_length

            # Parse form data
            fields = cgi.parse_multipart(self.rfile, pdict)

            # Debug print the entire fields dictionary
            print(fields)

            # Verify required fields
            if 'image' not in fields or 'points[]' not in fields or 'labels[]' not in fields:
                self.send_response(400)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(b"Invalid input: Missing image, coordinates, or labels.")
                return

            # Extract image data in binary form
            image_data = fields['image'][0]
            print('image_data is binary' if isinstance(image_data, bytes) else "image_data is not binary")

            # Extract and process coordinates and labels
            points = fields['points[]']
            coords_data = [list(map(int, point.split(','))) for point in points]
            labels = fields['labels[]']
            labels_data = [int(label) for label in labels]

            # Ensure image_data is bytes
            if isinstance(image_data, bytes):
                # Handle the image (open with PIL)
                try:
                    image = Image.open(io.BytesIO(image_data))
                    print("Successfully decoded image")
                except Exception as e:
                    self.send_response(400)
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(b"Error processing image.")
                    return

                # Handle coordinates (convert to a NumPy array)
                try:
                    points_array = np.array(coords_data)
                    print(points_array)
                    print("Successfully decoded coordinates")
                except Exception as e:
                    self.send_response(400)
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(b"Error processing coordinates.")
                    return

                # Handle labels (convert to a NumPy array)
                try:
                    labels_array = np.array(labels_data)
                    print(labels_array)
                    print("Successfully decoded labels")
                except Exception as e:
                    self.send_response(400)
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(b"Error processing labels.")
                    return

                try:
                    masked_image = apply_sam2_mask(image, points_array, labels_array)
                    # Prepare response
                    buffered = io.BytesIO()
                    masked_image.save(buffered, format="PNG")
                    response_data = buffered.getvalue()
                    
                    # Send response
                    self.send_response(200)
                    self._set_cors_headers()
                    self.send_header("Content-type", "image/png")
                    self.send_header("Content-Length", str(len(response_data)))
                    self.end_headers()
                    self.wfile.write(response_data)
                    print("Sent result masked image to client")
                except Exception as e:
                    self.send_response(500)
                    self._set_cors_headers()
                    self.end_headers()
                    self.wfile.write(b"Error processing the mask.")
                    return
            else:
                self.send_response(400)
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(b"Image data is not in correct binary format.")
        else:
            self.send_response(400)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(b"Unsupported content type.")

def run(server_class=HTTPServer, handler_class=SimpleCORSRequestHandler):
    server_address = ('', PORT)
    httpd = server_class(server_address, handler_class)
    print(f"Serving on port {PORT}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()