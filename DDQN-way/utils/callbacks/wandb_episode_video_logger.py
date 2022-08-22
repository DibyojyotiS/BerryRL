import os
import wandb
from utils.visualization import picture_episodes

class wandbEpisodeVideoLogger:
    def __init__(
        self, log_dir:str, save_dir:str, 
        train_log_freq=100, eval_log_freq=10, figsize=(20,20),
        n_parallelize=5, fps=1, logging=True, syncing=False
    ) -> None:
        """ 
        NOTE:
            the working of this function assumes that the info-dict's
            direct keys may eval if evaluation is enabled, this is used to
            assert wether or not to try forming the evaluation video.
            
            This function assumes that logging is being done by the berry-field-env"""
        assert logging or syncing
        self.log_dir = log_dir
        self.save_dir = save_dir
        self.train_log_freq = train_log_freq
        self.eval_log_freq = eval_log_freq
        self.n_parallelize = n_parallelize
        self.fps = fps
        self.figsize = figsize

        self.logging = logging
        self.syncing = syncing

        self.train_steps = 1
        self.eval_steps = 1
        self.last_train_episode = 0
        self.last_eval_episode = 0

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(self, info_dict:dict):
        
        video_log = {}

        if "train" in info_dict:
            if self.train_steps % self.train_log_freq == 0:
                self.train_steps = 1

                # make continue video/gif and send to wandb
                current_episode = info_dict["train"]["trainEpisode"]
                video_fn = f"train-episodes-{self.last_train_episode}-{current_episode}.mp4"
                video_fp = os.path.join(self.save_dir, video_fn)
                picture_episodes(
                    fname=video_fp, LOG_DIR=self.log_dir,
                    episodes=range(self.last_train_episode, current_episode+1),
                    figsize=self.figsize, titlefmt='train-episode {}',
                    nparallel=self.n_parallelize, fps=self.fps
                )
                if self.logging:
                    train_video= wandb.Video(
                        data_or_path=video_fp, format='mp4',
                        caption= video_fn, fps=self.fps
                    )
                    self.last_train_episode = current_episode+1
                    video_log["train"] = {"video": train_video}
                if self.syncing: 
                    wandb.save(video_fp)
            else:
                self.train_steps += 1

        if "eval" in info_dict:
            if self.eval_steps % self.eval_log_freq == 0:
                self.eval_steps = 1
                current_episode = self.last_eval_episode + self.eval_log_freq - 1
                eval_log_dir = os.path.join(self.log_dir, 'eval')
                video_fn = f"eval-episodes-{self.last_eval_episode}-{current_episode}.mp4"
                video_fp = os.path.join(self.save_dir, video_fn)
                picture_episodes(
                    fname=video_fp, LOG_DIR=eval_log_dir,
                    episodes=range(self.last_eval_episode, current_episode+1),
                    figsize=self.figsize, titlefmt='eval-episode {}', 
                    nparallel=self.n_parallelize, fps=self.fps
                )
                if self.logging:
                    eval_video = wandb.Video(
                        data_or_path=video_fp, format='mp4',
                        caption= video_fn, fps=self.fps
                    )
                    video_log["eval"] = {"video": eval_video}
                if self.syncing: 
                    wandb.save(video_fp, base_path=self.save_dir)
            else:
                self.eval_steps += 1

        if video_log:
            print(f"Uploading video-log for {[*video_log.keys()]}")
            wandb.log(video_log, commit=False)
