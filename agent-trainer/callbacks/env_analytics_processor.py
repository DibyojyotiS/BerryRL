import os
from wandb import Image, Video
from os.path import join, exists
from berry_field import BerryFieldEnv
from berry_field.envs.analysis.visualization import picture_episode, juice_plot
from .util_video_maker import VideoMaker

class JuicePlotter:
    def __init__(self, save_dir, max_steps) -> None:
        self.save_dir = save_dir
        self.max_steps = max_steps

        if not exists(self.save_dir):
           os.makedirs(self.save_dir) 
    
    def save_pic(self, title, analysis_dict):
        save_path=join(self.save_dir, f"{title}.png")
        juice_plot(
            sampled_juice=analysis_dict["sampled_juice"],
            total_steps=analysis_dict["env_steps"],
            max_steps=self.max_steps,
            title=title,
            nberries_picked=analysis_dict["berries_picked"],
            savepth=save_path,
            show=False,
            close=True
        )
        return save_path


class FieldPlotter:
    def __init__(self, save_dir, field_size) -> None:
        self.save_dir = save_dir
        self.field_size = field_size

        if not exists(self.save_dir):
           os.makedirs(self.save_dir) 

    def save_pic(self, title, analysis_dict):
        """ returns the path where plt plot is saved"""
        return self._save_field_pic(
            analysis_dict=analysis_dict,
            plot_title=title
        )

    def _save_field_pic(self, analysis_dict, plot_title):
        save_path=join(self.save_dir, f"{plot_title}.png")
        self._make_and_save_episode_picture(
                field_size=self.field_size,
                analysis_info=analysis_dict,
                title=plot_title,
                save_path=save_path
            )
        return save_path

    def _make_and_save_episode_picture(
        self, field_size, analysis_info, title, save_path
    ):
        picture_episode(
            berry_boxes=analysis_info["berry_boxes"],
            patch_boxes=analysis_info["patch_boxes"],
            sampled_path=analysis_info["sampled_path"],
            nberries_picked=analysis_info["berries_picked"],
            total_steps=analysis_info["env_steps"],
            field_size=field_size,
            title=title,
            show=False,
            savepth=save_path,
            close=True
        )


class BerryFieldAnalyticsProcessor:
    """ cleans/extract and appends the stats into 
    info-dict for direct logging. Also create and 
    save some plots (and videos of those plots).
    Only videos are logged to wandb """
    def __init__(
        self, 
        berry_field:BerryFieldEnv,
        save_dir:str,
        wandb:bool,
        episodes_per_video:int
    ) -> None:
        self.berry_env = berry_field
        self.wandb_enabled = wandb
        self.episodes_per_video = episodes_per_video
        self.episode = 0
        self._init_plotting(save_dir)

    def __call__(self, info:dict):
        field_video = self._make_field_plot_and_video(info)
        juice_video = self._make_juice_plot_and_video(info)

        info["env"] = self._extract_infos(info)
        if juice_video is not None:
            info["env"]["juice_plot"] = field_video
        if field_video is not None:
            info["env"]["field_plot"] = juice_video
            
        self.episode += 1
        return info

    def _make_juice_plot_and_video(self, info):
        analysis_dict = info["raw_stats"]["env"]
        self.juice_plots_list.append(
            self.juice_plotter.save_pic(
                title=f"episode-{self.episode}",
                analysis_dict=analysis_dict
            )
        )
        if len(self.juice_plots_list) >= self.episodes_per_video:
            file_name = self.juice_videomaker.create_video(self.juice_plots_list)
            self.juice_plots_list.clear()
            if self.wandb_enabled: return Video(file_name)
        return None

    def _make_field_plot_and_video(self, info):
        analysis_dict = info["raw_stats"]["env"]
        self.field_plots_list.append(
            self.field_plotter.save_pic(
                title=f"episode-{self.episode}",
                analysis_dict=analysis_dict
            )
        )
        if len(self.field_plots_list) >= self.episodes_per_video:
            file_name = self.field_videomaker.create_video(self.field_plots_list)
            self.field_plots_list.clear()
            if self.wandb_enabled: return Video(file_name)
        return None

    def _extract_infos(self, info:dict):
        analysis_dict:dict = info["raw_stats"]["env"]
        extracted =  {
            "berry_preferences": analysis_dict["berry_preferences"],
            "berries_picked": analysis_dict["berries_picked"],
            "num_visited_patches": len(analysis_dict["visited_patches"]),
            "env_steps": analysis_dict["env_steps"],
            "total_patch_central_time": \
                analysis_dict["total_patch_central_time"],
            "total_patch_periphery_time": \
                analysis_dict["total_patch_periphery_time"],
            "inter_patch_time": analysis_dict["inter_patch_time"],
        }
        for key in extracted.keys():
            analysis_dict.pop(key, False) # don't keep duplicates
        return extracted

    def _init_plotting(self, save_dir):
        self.field_plotter = FieldPlotter(
            save_dir=join(save_dir, "env-field"),
            field_size=self.berry_env.FIELD_SIZE
        )
        self.field_plots_list = []
        self.juice_plotter = JuicePlotter(
            save_dir=join(save_dir, "env-juice"),
            max_steps=self.berry_env.MAX_STEPS
            # divide by 10 to match decimation
        )
        self.juice_plots_list = []
        self.juice_videomaker = VideoMaker(
            save_dir=join(save_dir, "env-juice-video"),
            fps=1
        )
        self.field_videomaker = VideoMaker(
            save_dir=join(save_dir, "env-field-video"),
            fps=1
        )
