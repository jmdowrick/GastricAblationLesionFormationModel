FROM dolfinx/dolfinx:v0.11.0

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
