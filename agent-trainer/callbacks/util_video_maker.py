import imageio
import os
from os.path import exists


class VideoMaker:
    """ makes mp4 video from a list of images provided as file-names """
    def __init__(self, save_dir, fps=1) -> None:
        self.fps = fps
        self.save_dir = save_dir
        if not exists(self.save_dir):
           os.makedirs(self.save_dir) 

    def create_video(self, list_of_files) -> str:
        path = self._create_fpath(list_of_files)
        print("saving video to", path)
        with imageio.get_writer(path, fps=self.fps) as f:
            for plot_file in list_of_files:
                img = imageio.imread(plot_file)
                f.append_data(img)
        return path

    def _create_fpath(self, list_of_files):
        start:str = os.path.split(list_of_files[0])[-1]
        end:str = os.path.split(list_of_files[-1])[-1]
        fname = (
            ''.join(start.split('.')[:-1])
            + '-'
            + ''.join(end.split('.')[:-1])
            + '.mp4'
        )
        return os.path.join(self.save_dir, fname)