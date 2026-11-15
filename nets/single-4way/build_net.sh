#!/usr/bin/env bash
# Rebuild intersection.net.xml from the plain-XML sources (only needed if you change them).
#   nodes: 4 approaches, 250 m long; edges: 3 lanes each
#   connections: lane 0 right+straight, lane 1 straight, lane 2 left
#   traffic light: 4 phases (NS through, NS left, EW through, EW left) + 3 s yellows
cd "$(dirname "$0")"
netconvert -n intersection.nod.xml -e intersection.edg.xml -x intersection.con.xml \
  -i intersection.tll.xml -o intersection.net.xml --no-turnarounds true
