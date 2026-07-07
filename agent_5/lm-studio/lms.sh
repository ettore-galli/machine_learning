#!/bin/bash
export SERVER_PORT="9876"
export SERVER_HOST="localhost"

lms server start --port "${SERVER_PORT}" --cors --bind "${SERVER_HOST}"