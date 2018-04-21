# coding=utf-8
"""Performs face detection in realtime.

Based on code from https://github.com/shanren7/real_time_face_recognition
"""
# MIT License
#
# Copyright (c) 2017 François Gervais
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
import sys
sys.path.append("../src/")
import time

import cv2

import face

from queue import Queue, Empty
from threading import Thread, Lock


def add_overlays(frame, faces, frame_rate):
    if faces is not None:
        for face in faces:
            face_bb = face.bounding_box.astype(int)
            cv2.rectangle(frame,
                          (face_bb[0], face_bb[1]), (face_bb[2], face_bb[3]),
                          (0, 255, 0), 2)
            if face.name is not None:
                cv2.putText(frame, face.name, (face_bb[0], face_bb[3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                            thickness=2, lineType=2)

    cv2.putText(frame, str(frame_rate) + " fps", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                thickness=2, lineType=2)

def detect_faces(recogniser, frames_s, faces_q, stack_lock, num_workers):
    while True:
        # with stack_lock:
        stack_lock.acquire()
        try:
            frame = frames_s.pop()
        except IndexError:
            continue
        finally:
            stack_size = len(frames_s)
            if stack_size > num_workers:
                del frames_s[: stack_size - num_workers]
            stack_lock.release()
        faces = recogniser.identify(frame)
        faces_q.put(faces, False)


def main(args):
    fps_display_interval = 1  # seconds
    frame_rate = 0
    frame_count = 0

    video_capture = cv2.VideoCapture(0)
    face_recognition = face.Recognition()
    start_time = time.time()

    if args.debug:
        print("Debug enabled")
        face.debug = True

    faces_q = Queue()
    frames_s = []
    stack_lock = Lock()

    num_workers = 2

    for _ in range(num_workers):
        t = Thread(target=detect_faces, kwargs={
            'recogniser': face_recognition,
            'frames_s': frames_s,
            'faces_q': faces_q,
            'stack_lock': stack_lock,
            'num_workers': num_workers,
        })
        t.start()

    faces = None

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        frames_s.append(frame)
        try:
            with faces_q.mutex:
                faces = faces_q.queue.pop()
                if (len(faces) > 1):
                    del faces[: len(faces) - 1]
            # Check our current fps
            frame_count += 1
            end_time = time.time()
            if (end_time - start_time) > fps_display_interval:
                frame_rate = int(frame_count / (end_time - start_time))
                start_time = time.time()
                frame_count = 0
        except (Empty, IndexError):
            pass   


        add_overlays(frame, faces, frame_rate)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true',
                        help='Enable some debug outputs.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
