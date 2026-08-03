# https://ollama.com/download
#
source ./env.sh

# PROMPT="Write a simple web page displaying 'hello world. Single file, keep things basic" 
PROMPT="Write a python function to calculate the mandelbrot set, returning a 100x100 matrix containing bailout values" 

ollama run ${OLLAMA_MODEL_NAME} ${PROMPT} --verbose

 