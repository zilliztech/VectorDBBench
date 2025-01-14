import argparse
import os
import subprocess


def docker_tag_base():
    return 'vdbbench'

def dockerfile_path_base():
    return os.path.join('vectordb_bench/', '../Dockerfile')

def docker_tag(track, algo):
    return docker_tag_base() + '-' + track + '-' + algo


def build(tag, args, dockerfile):
    print('Building %s...' % tag)
    if args is not None and len(args) != 0:
        q = " ".join(["--build-arg " + x.replace(" ", "\\ ") for x in args])
    else:
        q = ""

    try:
        command = 'docker build %s --rm -t %s -f' \
                   % (q, tag)
        command += ' %s .' % dockerfile
        print(command)
        subprocess.check_call(command, shell=True)
        return {tag: 'success'}
    except subprocess.CalledProcessError:
        return {tag: 'fail'}

def build_multiprocess(args):
    return build(*args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--proc",
        default=1,
        type=int,
        help="the number of process to build docker images")
    parser.add_argument(
        '--track',
        choices=['none'],
        default='none'
    )
    parser.add_argument(
        '--algorithm',
        metavar='NAME',
        help='build only the named algorithm image',
        default=None)
    parser.add_argument(
        '--dockerfile',
        metavar='PATH',
        help='build only the image from a Dockerfile path',
        default=None)
    parser.add_argument(
        '--build-arg',
        help='pass given args to all docker builds',
        nargs="+")
    args = parser.parse_args()

    print('Building base image...')

    subprocess.check_call(
        'docker build \
        --rm -t %s -f %s .' % (docker_tag_base(), dockerfile_path_base()), shell=True)

    print('Building end.')

