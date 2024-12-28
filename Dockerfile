FROM python:3.9-slim

# Install necessary tools and libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgsl-dev \
    libomp-dev \
    python3-dev \
    gcc \
    g++ \
    && dpkg -l | grep gsl \
    && ls /usr/include/gsl \
    && ls /usr/lib/aarch64-linux-gnu/libgsl* \
    && rm -rf /var/lib/apt/lists/*


# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && pip install cython numpy

# Set working directory
WORKDIR /work

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the source code
COPY src /work/src

# Build and install the Cython module
WORKDIR /work/src
RUN python setup.py clean --all && python setup.py build_ext --inplace
RUN pip install .

# Default command for debugging
CMD ["python", "-c", "print('Docker container ready!')"]
