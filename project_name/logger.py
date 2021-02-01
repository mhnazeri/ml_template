import numpy as np
from typing import Union, List, Optional, Dict
import torch
from omegaconf import DictConfig


class Logger:
    def __init__(
        self,
        project: str,
        name: str,
        tags,
        config,
        type_: str = "tensorboard",
        resume: bool = False,
        online: bool = True,
        experiment_id: Optional[str] = None,
        log_dir: str = "./log",
    ):
        """Factory class for different logging libraries

        Args:
            project: (str) project name
            name: (str) run name
            tags: (List[str]) list of tags to better group different runs
            config: (omegaconf.DictConfig) config dictionary
            type_: (str) which logging library to use as backend, it can be either `tensorboard`, `wandb`, or `comet_ml`
            resume: (bool) either resume the training and logging or start fresh
            online: (bool) whether to store data on cloud or not
            log_dir: (str) directory in which the log files are stored
        """
        self.project = project
        self.name = name
        self.tags = tags
        self.config = config
        self.resume = resume
        self.log_dir = log_dir
        self.online = online
        self.experiment_id = experiment_id
        self.logger = self._get_logger(type_.lower())

    def _get_logger(self, type_):
        if type_ == "tensorboard":
            return TensorboardLogger(self.log_dir)
        elif type_ == "wandb":
            return WandbLogger(
                project=self.project,
                name=self.name,
                tags=self.tags,
                config=self.config,
                resume=self.resume,
                id=self.experiment_id,
                log_dir=self.log_dir,
            )
        elif type_ == "comet_ml":
            return CometLogger(
                project_name=self.name,
                workspace=self.project,
                tags=self.tags,
                config=self.config,
                resume=self.resume,
                online=self.online,
                id=self.experiment_id,
                log_dir=self.log_dir,
            )
        else:
            raise ValueError(
                "Unknown parameter for logger type, it should be one of 'tensorboard', 'wandb', or 'comet_ml'"
            )


class CometLogger:
    def __init__(
        self,
        project: str,
        workspace: str,
        tags: List[str],
        config: DictConfig = None,
        online: bool = True,
        resume: Optional[bool] = False,
        id: Optional[str] = None,
        log_dir: Optional[str] = None,
    ) -> None:
        from comet_ml import (
            Experiment,
            ExistingExperiment,
            OfflineExperiment,
            ExistingOfflineExperiment,
        )

        if online and resume:
            if id:
                self.comet = ExistingExperiment(
                    previous_experiment=id, log_env_details=True, log_env_gpu=True
                )
            else:
                raise ValueError(
                    "You must provide the previous experiment id to continue"
                )
        elif online and not resume:
            self.comet = Experiment(
                project_name=project,
                workspace=workspace,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
            )

        elif not online and resume:
            if id:
                self.comet = ExistingOfflineExperiment(
                    previous_experiment=id,
                    offline_directory=log_dir,
                    log_env_details=True,
                    log_env_gpu=True,
                )
            else:
                raise ValueError(
                    "You must provide the previous experiment id to cpntinue"
                )
        elif not online and not resume:
            self.comet = OfflineExperiment(
                project_name=project,
                workspace=workspace,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
                offline_directory=log_dir,
            )

        self.comet.add_tags(tags)
        self.comet.log_parameters(config)

    # def log_metric(
    #     self,
    #     name: str,
    #     value: Union[float, int, bool, str],
    #     step: Optional[int] = None,
    #     epoch: Optional[int] = None,
    #     include_context: Optional[bool] = True,
    # ):
    #     self.comet.log_metric(
    #         name=name,
    #         value=value,
    #         step=step,
    #         epoch=epoch,
    #         include_context=include_context,
    #     )

    def log_metric(
        self,
        dic: Dict,
        prefix: Optional = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        self.comet.log_metrics(dic=dic, prefix=prefix, step=step, epoch=epoch)

    def log_image(
        self,
        image_data: Union[str, np.array, torch.Tensor],
        name: Optional[str] = None,
        overwrite: Optional[bool] = False,
        image_format: Optional[str] = "png",
        image_channels: Optional[str] = "last",
        step: Optional[int] = None,
    ):
        self.comet.log_image(
            image_data=image_data,
            name=name,
            overwrite=overwrite,
            image_format=image_format,
            image_channels=image_channels,
            step=step,
        )

    def log_seg_mask(
        self,
        imgs: torch.Tensor,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        class_labels: Dict,
    ):
        """Log a batch of images with their predicted and ground-truth masks.

        Args:
            imgs: (torch.Tensor) actual input images
            predictions: (torch.Tensor) predicted masks
            ground_truth: (torch.Tensor) ground-truth masks
            class_labels: (Dict) a dictionary assigning an index to each class
        """
        assert len(img.size()) == 4  # img should a batch of images
        data = {}

        for idx, img in enumerate(imgs):
            data[img_id].append(
                {
                    "input": img.cpu().detach().numpy(),
                    "predictions": {
                        "mask_data": predictions[img_id].cpu().detach().numpy(),
                        "class_labels": class_labels,
                    },
                    "ground_truth": {
                        "mask_data": ground_truth[img_id].cpu().detach().numpy(),
                        "class_labels": class_labels,
                    },
                }
            )

        self.comet.log_asset_data(data)

    def log_audio(
        self,
        audio: Union[np.array, str],
        file_name: str,
        sample_rate: int = 44100,
        metadata: dict = None,
        step: Union[int, None] = None,
    ):
        self.comet.log_audio(
            audio_data=audio,
            sample_rate=sample_rate,
            file_name=file_name,
            metadata=metadata,
            step=step,
        )

    def log_code(
        self, file_name: Union[str, None] = None, folder: Union[str, None] = None
    ):
        self.comet.log_code(file_name=file_name, folder=folder)

    def log_text(
        self, text: str, step: Optional[int] = None, metadata: Optional[Dict] = None
    ):
        self.comet.log_text(text=text, step=step, metadata=metadata)

    def send_notification(
        self, title: str, status: str = None, additional_data: Dict = None
    ):
        self.comet.send_notification(
            title=title, status=status, additional_data=additional_data
        )

    def set_epoch(self, epoch: int):
        self.comet.set_epoch(epoch=epoch)

    def set_step(self, step: int):
        self.comet.set_step(step=step)

    def log_video(self, tag, video: Union[np.array, Path]):
        raise NotImplementedError

    def log_pr_curve(self):
        raise NotImplementedError

    def log_line(self):
        raise NotImplementedError

    def log_scatter(self):
        raise NotImplementedError

    def log_bar(self):
        raise NotImplementedError

    def log_histogram(self, tag: str, array: Union[np.array, list], bins: int = 64):
        raise NotImplementedError

    def log_histogram_3d(
        self,
        values: Union[List, Tuple, np.array],
        name: Optional[str] = None,
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        metadata: Optional[Dict] = None,
    ):
        self.comet.log_histogram_3d(
            values=values, name=name, step=step, epoch=epoch, metadata=metadata
        )

    def log_plot_histogram(self):
        raise NotImplementedError

    def log_roc(self):
        not NotImplementedError

    def log_confusion_matrix(
        self,
        ground_truth,
        predictions,
        class_names: List[str] = None,
        matrix: Optional[List[List]] = None,
        title: str = "Confusion Matrix",
        row_label: str = "Actual Category",
        column_label: str = "Predicted Category",
        file_name: str = "confusion-matrix.json",
        step: Optional[int] = None,
        epoch: Optional[int] = None,
        **kwargs
    ):
        self.comet.log_confusion_matrix(
            y_true=ground_truth,
            y_predicted=predictions,
            labels=class_names,
            matrix=matrix,
            title=title,
            row_label=row_label,
            column_label=column_label,
            file_name=file_name,
            step=step,
            epoch=epoch,
        )

    def log_3d_object(self, tag: str, point_cloud: Union[np.array, str]):
        raise NotImplementedError

    def log_parameters(
        self, parameters: Dict, prefix: Optional = None, step: Optional[int] = None
    ):
        self.comet.log_parameters(parameters=parameters, prefix=prefix, step=step)

    def log_molecule(self, tag, file_path: str):
        raise NotImplementedError

    def log_html(self, html: str, clear: bool = False):
        self.comet.log_html(html=html, clear=clear)

    def log_curve(
        self,
        name: str,
        x: List,
        y: List,
        overwrite: Optional[bool] = False,
        step: Optional[int] = None,
    ):
        self.comet.log_curve(name=name, x=x, y=y, overwrite=overwrite, step=step)

    def log_dataset_info(
        self, name: Optional[str], version: Optional[str], path: Optional[str]
    ):
        self.comet.log_dataset_info(name=name, version=version, path=path)

    def log_epoch_end(self, epoch_cnt: int, step: Optional[int]) -> NoReturn:
        self.comet.log_epoch_end(epoch_cnt=epoch_cnt, step=step)

    def log_figure(
        self,
        figure_name: Optional[str],
        figure: Optional,
        overwrite: Optional[bool] = None,
        step: Optional[int] = None,
    ):
        """Logs the global Pyplot figure or the passed one and upload its svg version to the backend."""
        self.comet.log_figure(
            figure_name=figure_name, figure=figure, overwrite=overwrite, step=step
        )

    def log_model(
        self,
        name: str,
        file_or_folder: Union[str, object],
        file_name: Optional[str] = None,
        overwrite: Optional[bool] = True,
    ):
        self.comet.log_model(
            name=name,
            file_or_folder=file_or_folder,
            file_name=file_name,
            overwrite=overwrite,
        )

    def log_system_info(
        self, key: Union[str, int, float], value: Union[str, int, float]
    ):
        self.comet.log_system_info(key, value)

    def log_table(
        self,
        filename: str,
        tabular_data: Optional = None,
        headers: Union[bool, List] = False,
    ):
        self.comet.log_table(
            filename=filename, tabular_data=tabular_data, headers=headers
        )

    def log_embedding(self):
        raise NotImplementedError

    def export_scalars_to_json(self):
        raise NotImplementedError

    def log_asset_data(
        self,
        data,
        name: Optional[str] = None,
        overwrite: Optional[bool] = False,
        step: Optional[int] = None,
        metadata: Optional[Dict] = None,
        file_name: Optional[str] = None,
        epoch: Optional[int] = None,
    ):
        self.comet.log_asset_data(
            data=data,
            name=name,
            overwrite=overwrite,
            step=step,
            metadata=metadata,
            file_name=file_name,
            epoch=epoch,
        )

    def finalize(self):
        self.comet.clean()
        self.comet.end()

    def log_bbox(self, imgs, predictions, labels, label_map):
        data = {}

        for idx, img in enumerate(imgs):
            prediction = predictions[idx]
            label = labels[idx]

            predicted_boxes = prediction["boxes"].cpu().detach().numpy().tolist()
            predicted_scores = prediction["scores"].cpu().detach().numpy().tolist()
            predicted_classes = prediction["labels"].cpu().detach().numpy().tolist()

            label_boxes = label["boxes"].cpu().detach().numpy().tolist()

            data.setdefault(idx, [])
            self.log_image(img)
            for label_box in label_boxes:
                x, y, x2, y2 = label_box
                data[idx].append(
                    {
                        "label": "ground-truth",
                        "score": 100,
                        "box": {"x": x, "y": y, "x2": x2, "y2": y2},
                    }
                )

            for predicted_box, predicted_score, predicted_class in zip(
                predicted_boxes, predicted_scores, predicted_classes
            ):
                x, y, x2, y2 = predicted_box
                data[idx].append(
                    {
                        "label": label_map[predicted_class - 1],
                        "box": {"x": x, "y": y, "x2": x2, "y2": y2},
                        "score": predicted_score * 100,
                    }
                )

        self.comet.log_asset_data(data)


class WandbLogger:
    def __init__(
        self,
        project: str,
        name: str,
        tags,
        config: DictConfig,
        resume: bool,
        id,
        log_dir,
    ):
        import wandb  # only install wandb if needed

        self.wb = wandb.init(
            project=project, name=name, tags=tags, config=config, resume=resume, id=id
        )
        self.wb.run.dir = log_dir

    def log_metric(self, metric: dict):
        self.wb.log(metric)

    def log_image(self, tag: str, img: np.array, caption: str):
        self.wb.log({tag: [wandb.Image(img, caption=caption)]})

    def log_model(self, model):
        self.wb.watch(model)

    def log_seg_mask(
        self,
        tag: str,
        img: torch.Tensor,
        predictions: torch.Tensor,
        ground_truth: torch.Tensor,
        class_labels: dict,
    ):
        """Log a batch of images with their predicted and ground-truth masks.

        Args:
            tag: (str) Caption for the panel
            img: (torch.Tensor) actual input images
            predictions: (torch.Tensor) predicted masks
            ground_truth: (torch.Tensor) ground-truth masks
            class_labels: (dict) a dictionary assigning an index to each class
        """
        assert len(img.size()) == 4  # img should a batch of images
        self.wb.log(
            {
                tag: [
                    wandb.Image(
                        img[i].cpu().detach().permute(1, 2, 0).numpy(),
                        masks={
                            "predictions": {
                                "mask_data": predictions[i],
                                "class_labels": class_labels,
                            },
                            "ground_truth": {
                                "mask_data": ground_truth[i].cpu().detach().numpy(),
                                "class_labels": class_labels,
                            },
                        },
                    )
                    for i in range(img.size(0))
                ]
            }
        )

    def log_audio(
        self, tag: str, audio: np.array, caption: str, sample_rate: int = 44100
    ):
        self.wb.log(
            {tag: [wandb.Audio(audio, caption=caption, sample_rate=sample_rate)]}
        )

    def log_text(
        self,
        tag: str,
        text_list: List[str],
        columns: List[str] = ["Text", "Predicted Label", "True Label"],
    ):
        self.wb.log({tag: wandb.Table(data=text_list, columns=columns)})

    def log_video(self, tag, video: Union[np.array, Path]):
        self.wb.log(
            {tag: wandb.Video(numpy_array_or_path_to_video, fps=4, format="gif")}
        )

    def log_pr_curve(
        self,
        tag: str,
        ground_truth,
        predictions,
        labels: List[str] = None,
        classes_to_plot=None,
    ):
        self.wb.log(
            {
                tag: wandb.plot.pr_curve(
                    ground_truth,
                    predictions,
                    labels=labels,
                    classes_to_plot=classes_to_plot,
                )
            }
        )

    def log_line(
        self,
        tag: str,
        data: List[List],
        columns: List[str] = ["x", "y"],
        title: str = "Custom Y vs X Line Plot",
    ):
        table = wandb.Table(data=data, columns=columns)
        self.wb.log({tag: wandb.plot.line(table, *columns, title=title)})

    def log_scatter(
        self, tag: str, data: List[List], columns: List[str] = ["class_x", "class_y"]
    ):
        table = wandb.Table(data=data, columns=columns)
        self.wb.log({tag: wandb.plot.scatter(table, *columns)})

    def log_bar(
        self,
        tag: str,
        data: List[List],
        columns: List[str] = ["label", "value"],
        title: str = "Custom Bar Chart",
    ):
        table = wandb.Table(data=data, columns=columns)
        self.wb.log({tag: wandb.plot.bar(table, *columns, title=title)})

    def log_histogram(self, tag: str, array: Union[np.array, list], bins: int = 64):
        assert bins <= 512
        self.wb.log({tag: wandb.Histogram(array, num_bins=bins)})

    def log_plot_histogram(
        self,
        tag: str,
        data: List,
        columns: List[str] = ["value"],
        title: str = "Custom Histogram",
    ):
        data = [[d] for d in data]
        table = wandb.Table(data=data, columns=columns)
        self.wb.log({tag: wandb.plot.histogram(table, *columns, title=title)})

    def log_roc(
        self,
        tag: str,
        ground_truth,
        predictions,
        labels: List[str] = None,
        classes_to_plot=None,
    ):
        self.wb.log(
            {
                tag: wandb.plot.roc_curve(
                    ground_truth,
                    predictions,
                    labels=labels,
                    classes_to_plot=classes_to_plot,
                )
            }
        )

    def log_confusion_matrix(
        self, tag: str, ground_truth, predictions, class_names: List[str] = None
    ):
        self.wb.log(
            {tag: wandb.plot.confusion_matrix(predictions, ground_truth, class_names)}
        )

    def log_3d_object(self, tag: str, point_cloud: Union[np.array, str]):
        if isinstance(point_cloud, str):
            assert point_cloud.split(".")[-1] in ["obj", "gltf", "glb"]
            self.wb.log({tag: wandb.Object3D(open(point_cloud))})
        elif isinstance(point_cloud, np.array):
            self.wb.log({tag: wandb.Object3D(point_cloud)})
        else:
            raise ValueError("unknown type for 3D object")

    def log_molecule(self, tag, file_path: str):
        self.wb.log({tag: wandb.Molecule(open(file_path))})

    def log_html(self, tag, file_path: str, inject: bool = False):
        self.wb.log({tag: wandb.Html(open(file_path), inject=inject)})

    def log_embedding(self):
        raise NotImplementedError

    def export_scalars_to_json(self):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError


class TensorboardLogger:
    def __init__(self, log_dir):
        from tensorboardX import SummaryWriter

        self.tb = SummaryWriter(log_dir)

    # def log_metric(self, tag: str, scalar_value, global_step=None, walltime=None, main_tag = "default"):
    #     self.tb.add_scalar(tag=tag, scalar_value=scalar_value, global_step=global_step, walltime=walltime, main_tag=main_tag)

    def log_metric(
        self, main_tag: str, tag_scalar_dict, global_step=None, walltime=None
    ):
        self.tb.add_scalar(
            main_tag=main_tag,
            tag_scalar_dict=tag_scalar_dict,
            global_step=global_step,
            walltime=walltime,
        )

    def log_image(
        self,
        tag: str,
        img,
        global_step: int = None,
        walltime=None,
        dataformats: str = "CHW",
    ):
        self.tb.add_image(
            tag=tag,
            img_tensor=img,
            global_step=global_step,
            walltime=walltime,
            dataformats=dataformats,
        )

    def log_audio(
        self,
        tag: str,
        audio,
        global_step: int = None,
        sample_rate: int = 44100,
        walltime=None,
    ):
        self.tb.add_audio(
            tag=tag,
            snd_tensor=audio,
            global_step=global_step,
            sample_rate=sample_rate,
            walltime=walltime,
        )

    def log_text(
        self, tag: str, text_string: str, global_step: int = None, walltime=None
    ):
        self.tb.add_image(
            tag=tag, text_string=text_string, global_step=global_step, walltime=walltime
        )

    def log_pr_curve(
        self,
        tag: str,
        ground_truth,
        predictions,
        global_step=None,
        num_thresholds: int = 127,
        weights=None,
    ):
        self.tb.add_image(
            tag=tag,
            labels=ground_truth,
            predictions=predictions,
            global_step=global_step,
            num_thresholds=num_thresholds,
            weights=weights,
        )

    def log_histogram(
        self,
        tag: str,
        array,
        global_step=None,
        bins: str = "tensorflow",
        walltime=None,
        max_bins=None,
    ):
        self.tb.add_image(
            tag=tag,
            values=array,
            global_step=global_step,
            bins=bins,
            walltime=walltime,
            max_bins=max_bins,
        )

    def log_embedding(
        self,
        mat,
        metadata=None,
        label_img=None,
        global_step=None,
        tag="default",
        metadata_header=None,
    ):
        self.tb.add_image(
            mat=mat,
            metadata=metadata,
            label_img=label_img,
            global_step=global_step,
            tag=tag,
            metadata_header=metadata_header,
        )

    def export_scalars_to_json(self, path):
        self.tb.export_scalars_to_json(path)

    def finalize(self):
        self.tb.close()

    def log_seg_mask(self):
        raise NotImplementedError

    def log_video(self):
        raise NotImplementedError

    def log_molecule(self, tag, file_path: str):
        raise NotImplementedError

    def log_html(self, tag, file_path: str, inject: bool = False):
        raise NotImplementedError

    def log_line():
        raise NotImplementedError

    def log_bar():
        raise NotImplementedError

    def log_scatter():
        raise NotImplementedError
