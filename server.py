import http.server
import socketserver
import io
from PIL import Image
import numpy as np
import cgi

from run_image_predictor import apply_sam2_mask

PORT = 8002

class SimpleHTTPRequestHandler(http.server.BaseHTTPRequestHandler):

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
            if 'image' not in fields or 'points[]' not in fields:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Invalid input: Missing image or coordinates.")
                return

            # Extract image data in binary form
            image_data = fields['image'][0]
            print('image_data is binary' if isinstance(image_data, bytes) else "image_data is not binary")
            points = fields['points[]']
            coords_data = [list(map(int, point.split(','))) for point in points]  # Assuming points are sent as a comma-separated string
            labels = fields['labels[]']
            labels_data = [int(label) for label in labels]  # Assuming points are sent as a comma-separated string

            # Ensure image_data is bytes
            if isinstance(image_data, bytes):
                # Handle the image (open with PIL)
                try:
                    image = Image.open(io.BytesIO(image_data))
                    print("Successfully decoded image")
                except Exception as e:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Error processing image.")
                    return

                # Handle coordinates (convert to a NumPy array)
                try:
                    points = np.array(coords_data)
                    print (points)
                    print("Successfully decoded coordinates")
                except Exception as e:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Error processing coordinates.")
                    return

                # Handle lables (convert to a NumPy array)
                try:
                    labels = np.array(labels_data)
                    print (labels)
                    print("Successfully decoded labels")
                except Exception as e:
                    self.send_response(400)
                    self.end_headers()
                    self.wfile.write(b"Error processing labels.")
                    return

                masked_image = apply_sam2_mask(image, points, labels)
                # Prepare response
                buffered = io.BytesIO()
                masked_image.save(buffered, format="PNG")
                response_data = buffered.getvalue()

                # Send response
                self.send_response(200)
                self.send_header("Content-type", "image/png")
                self.send_header("Content-Length", str(len(response_data)))
                self.end_headers()
                self.wfile.write(response_data)
                print("Sent result masked image to client")


            else:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b"Image data is not in correct binary format.")
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Unsupported content type.")

def run(server_class=http.server.HTTPServer, handler_class=SimpleHTTPRequestHandler):
    server_address = ('', PORT)
    httpd = server_class(server_address, handler_class)
    print(f"Serving on port {PORT}")
    httpd.serve_forever()

if __name__ == "__main__":
    run()