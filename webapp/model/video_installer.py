from pytube import YouTube
from properties.common_property import *

def download(link):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    try:
        youtubeObject.download(VIDEO_INSTALLER_OUTPUT_PATH)
    except:
        print("An error has occurred")

    print("Download is completed successfully")

def getAllLink():
    video_link_list = []
    with open(VIDEO_INSTALLER_INPUT_PATH) as file:
        while True:
            line = file.readline()
            if not line:
                break
            video_link_list.append(line.strip())

    return video_link_list

def main():
    input = getAllLink()
    if input is not None:
        for link in input:
            download(link)


if __name__ == "__main__":
    main()
