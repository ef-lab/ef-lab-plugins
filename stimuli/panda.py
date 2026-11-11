# Derived from ef-lab-plugins (MIT). See LICENSE.
from copy import deepcopy
from itertools import groupby
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from urllib.parse import quote

import datajoint as dj
import numpy as np
import panda3d.core as core
from scipy import interpolate
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import CardMaker, NodePath, TextureStage
from direct.actor.Actor import Actor
from typing import Any, Dict

from ethopy import __version__
from ethopy.core.logger import stimulus
from ethopy.core.stimulus import (
    StimCondition,
    Stimulus,
)  # import StimCondition need for the Panda class definition
from ethopy.utils.helper_functions import DictStruct, iterable
from ethopy.utils.timer import Timer


@stimulus.schema
class Objects(dj.Lookup):
    definition = """
    # object information
    obj_id               : int                          # object ID
    ---
    description          : varchar(256)                 # description
    object=null          : longblob                     # 3d file
    file_name=null       : varchar(255)   
    source=null          : varchar(255) 
    type='static'        : enum('static','morph','motion')     
    """

    def store(self, key: Dict[str, Any], replace=False):
        key['object'] = np.fromfile(key['file_name'], dtype=np.int8)
        self.insert1(key, replace=replace)


@stimulus.schema
class Panda(Stimulus, dj.Manual):
    definition = """
    # This class handles the presentation of Objects with Panda3D
    -> StimCondition
    """

    class Object(dj.Part):
        definition = """
        # object conditions
        -> Panda
        -> Objects
        ---
        obj_pos_x             : blob
        obj_pos_y             : blob
        obj_mag               : blob
        obj_rot               : blob
        obj_tilt              : blob 
        obj_yaw               : blob
        obj_delay             : int
        obj_dur               : int
        obj_occluder          : int
        perspective           : int
        obj_animation=null        : blob
        """

    class Environment(dj.Part):
        definition = """
        # object conditions
        -> Panda
        ---
        background_color      : tinyblob
        ambient_color         : tinyblob
        """

    class Movie(dj.Part):
        definition = """
        # object conditions
        -> Panda
        ---
        movie_name            : char(8)                      # short movie title
        clip_number           : int                          # clip index
        """

    class Light(dj.Part):
        definition = """
        # object conditions
        -> Panda
        light_idx             : tinyint
        ---
        light_color           : tinyblob
        light_dir             : tinyblob
        """

    cond_tables = [
        "Panda",
        "Panda.Object",
        "Panda.Environment",
        "Panda.Light",
        "Panda.Movie",
    ]
    required_fields = ["obj_id", "obj_dur"]
    default_key = {
        "background_color": (0, 0, 0),
        "ambient_color": (0.1, 0.1, 0.1, 1),
        "light_idx": (1, 2),
        "light_color": (np.array([0.7, 0.7, 0.7, 1]), np.array([0.2, 0.2, 0.2, 1])),
        "light_dir": (np.array([0, -20, 0]), np.array([180, -20, 0])),
        "obj_pos_x": 0,
        "obj_pos_y": 0,
        "obj_mag": 0.5,
        "obj_rot": 0,
        "obj_tilt": 0,
        "obj_yaw": 0,
        "obj_delay": 0,
        "obj_occluder": 0,
        "perspective": 0,
        "obj_animation": None,
    }

    object_files = {}

    def __init__(self):
        super().__init__()
        self.is_recording = False
        self.record_format = None
        self._recording_ready = False
        self._recording_started = False
        self._recording_prepared = False
        self._logged_recording_session = None
        self._capture_dir = None
        self._capture_file = None
        self._encoder = None
        # DataJoint instantiates the table before the module binds Panda.
        self.cond_tables = __class__.cond_tables.copy()
        self.required_fields = __class__.required_fields.copy()
        self.default_key = deepcopy(__class__.default_key)

    def init(self, exp):
        if self.is_recording:
            # Match Stimulus.init's screen lookup without its Pi display/touch setup.
            self.exp, self.logger = exp, exp.logger
            screen = self.logger.get(
                schema="interface",
                table="SetupConfiguration.Screen",
                key=f"setup_conf_idx={exp.setup_conf_idx}",
                as_dict=True,
            )
            self.monitor = DictStruct(screen[0])
        else:
            super().init(exp)

        # Lazily mix ShowBase into this instance's class so the Panda3D window
        # only opens when the stimulus runs. Flag-guarded with _showbase_mixed
        # to avoid re-wrapping on repeated init() calls.
        cls = self.__class__
        if not getattr(cls, "_showbase_mixed", False):
            new_cls = cls.__class__(cls.__name__, (cls, ShowBase), {})
            new_cls._showbase_mixed = True
            self.__class__ = new_cls
        if not self.is_recording:
            self.fStartDirect = self.logger.is_pi
            self.windowType = None if self.logger.is_pi else "onscreen"
            self.Fullscreen = self.logger.is_pi

        self.path = (
            self.logger.source_path + "/objects/"
        )  # default path to copy local stimuli
        self.movie_path = self.logger.source_path + "/movies/"

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.movie_path, exist_ok=True)

        self.fill_colors.background_color = (0, 0, 0)

    def setup(self):
        if self.is_recording:
            self._setup_recording_output()
        else:
            self._setup_onscreen_output()

        self.disableMouse()
        self.in_operation = False
        self.present_movie = False

        self.ambientLight = core.AmbientLight("ambientLight")
        self.ambientLightNP = self.render.attachNewNode(self.ambientLight)
        self.render.setLight(self.ambientLightNP)
        if not self.is_recording:
            self.set_taskMgr()

    def prepare(self, curr_cond, stim_period="", *, duration_ms=None):
        if self.is_recording and self._recording_prepared:
            raise RuntimeError("Stop the previous recording period before preparing another")
        self.flag_no_stim = False
        if stim_period == "":
            self.curr_cond = curr_cond
        elif stim_period not in curr_cond:
            self.flag_no_stim = True
            return
        else:
            self.curr_cond = curr_cond[stim_period]

        self.period = (stim_period or "Trial") if self.is_recording else stim_period
        self.objects = {}
        self.lights = {}
        self.lightsNP = {}
        if self.is_recording:
            self._recording_duration_ms = duration_ms
            self._recording_prepared = True

        # set background color
        self.set_background_color(*self.curr_cond["background_color"])

        # Set Ambient Light
        self.ambientLight.setColor(self.curr_cond["ambient_color"])

        # Set Directional Light
        for idx, _ in enumerate(iterable(self.curr_cond["light_idx"])):
            self.lights[idx] = core.DirectionalLight("directionalLight_%d" % idx)
            self.lightsNP[idx] = self.render.attachNewNode(self.lights[idx])
            self.render.setLight(self.lightsNP[idx])
            self.lights[idx].setColor(tuple(self.curr_cond["light_color"][idx]))
            self.lightsNP[idx].setHpr(*self.curr_cond["light_dir"][idx])

        # Match Ethopy's row expansion for a sequence-valued object key.
        obj_ids = iterable(self.curr_cond["obj_id"])
        single_object = np.isscalar(self.curr_cond["obj_id"])
        for idx, _ in enumerate(obj_ids):
            cond = self.get_cond("obj_", None if single_object else idx)
            self.objects[idx] = Agent(self, cond)

        if "movie_name" in self.curr_cond:
            file_name = self.get_clip_info(self.curr_cond, "file_name")
            self.mov_texture = self.loader.loadTexture(self.movie_path + file_name[0])
            if self.is_recording:
                if not isinstance(self.mov_texture, core.MovieTexture):
                    raise RuntimeError("Recording with a movie background requires a Panda3D MovieTexture")
                self.mov_texture.stop()
            cm = CardMaker("card")
            tx_scale = self.mov_texture.getTexScale()
            cm.setFrame(-1, 1, -tx_scale[1] / tx_scale[0], tx_scale[1] / tx_scale[0])
            self.movie_node = NodePath(cm.generate())
            self.movie_node.setTexture(self.mov_texture, 1)
            self.movie_node.setPos(0, 100, 0)
            self.movie_node.setTexScale(
                TextureStage.getDefault(), self.mov_texture.getTexScale()
            )
            self.movie_node.setScale(48)
            self.movie_node.reparentTo(self.render)
            self.present_movie = True

    def start(self):
        if self.flag_no_stim:
            return
        if self.is_recording:
            self._open_capture()
        else:
            self.fill()
            if not self.in_operation:
                self.timer.start()
                self.in_operation = True

        self.log_start()
        if self.is_recording:
            self._recording_started = True
            for obj in self.objects.values():
                obj.load_model()
            self._capture_complete = self._frame_count == 0
            self.in_operation = not self._capture_complete
        else:
            if self.present_movie:
                self.mov_texture.play()
            for obj in self.objects.values():
                obj.run()
            self.flip(2)

    def present(self):
        if self.is_recording:
            if self.in_operation:
                self._capture_frame()
            return
        self.flip()
        if (
            "obj_dur" in self.curr_cond
            and np.max(self.curr_cond["obj_dur"]) < self.timer.elapsed_time()
        ):
            self.in_operation = False

    def flip(self, n=1):
        for i in range(0, n):
            self.taskMgr.step()

    def stop(self):
        if self.is_recording:
            self._stop_recording()
            return
        if self.flag_no_stim:
            return
        self._clear_scene()
        self.flip(2)  # clear double buffer
        self.log_stop()
        self.in_operation = False

    def record(self, format="npy", *, db_log=True, save_path=None, rec_aim="panda"):
        """Enable offscreen capture before make_conditions() initializes Panda."""
        format = format.lower()
        if format not in ("npy", "mov", "png"):
            raise ValueError("Panda.record() format must be npy, mov or png")
        if getattr(self, "exp", None) is not None:
            raise RuntimeError("Call Panda.record() before exp.make_conditions()")
        self.record_format = format
        self.recording_aim = rec_aim
        self.recording_db_log = db_log
        self.recording_path = (
            Path(save_path).expanduser().resolve() if save_path is not None else None
        )
        self.is_recording = True

    def _setup_onscreen_output(self):
        ShowBase.__init__(
            self, fStartDirect=self.fStartDirect, windowType=self.windowType
        )
        self.props = core.WindowProperties()
        if self.monitor.fullscreen:
            self.props.setSize(
                self.pipe.getDisplayWidth(), self.pipe.getDisplayHeight()
            )
            self.props.setFullscreen(self.Fullscreen)
        else:
            self.props.setSize(self.monitor.resolution_x, self.monitor.resolution_y)
        self.props.setCursorHidden(True)
        self.props.setUndecorated(True)
        self.win.requestProperties(self.props)
        self.graphicsEngine.openWindows()
        self.set_background_color(0, 0, 0)

    def _setup_recording_output(self):
        if self.recording_db_log:
            aim = self.logger.get(
                schema="recording",
                table="Aim",
                key={"rec_aim": self.recording_aim},
                fields=["rec_aim"],
            )
            if not len(aim):
                raise ValueError(f"Add recording aim {self.recording_aim!r} to recording.Aim first")
            software = self.logger.get(
                schema="recording",
                table="Software",
                key={"software": "EthoPy", "version": __version__},
                fields=["version"],
            )
            if not len(software):
                raise ValueError(f"Add EthoPy version {__version__!r} to recording.Software first")
        width, height = int(self.monitor.resolution_x), int(self.monitor.resolution_y)
        if width <= 0 or height <= 0:
            raise ValueError("Recording requires positive Screen resolution_x and resolution_y")
        self.fps = float(self.monitor.fps)
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Recording requires a positive SetupConfiguration.Screen fps")
        props = core.WindowProperties.size(width, height)
        prc = core.loadPrcFileData(
            "ethopy-panda-recording",
            "sync-video false\naudio-library-name null\nthreading-model /",
        )
        try:
            ShowBase.__init__(self, fStartDirect=False, windowType="none")
            try:
                if not self.openDefaultWindow(type="offscreen", props=props, startDirect=False):
                    raise RuntimeError("Panda3D could not open an offscreen buffer")
                if (self.win.getXSize(), self.win.getYSize()) != (width, height):
                    raise RuntimeError("Panda3D did not allocate the requested capture size")
                self.adjustWindowAspectRatio(width / height)
                self._capture_texture = core.Texture("ethopy-rgb")
                self.win.addRenderTexture(
                    self._capture_texture,
                    core.GraphicsOutput.RTMCopyRam,
                    core.GraphicsOutput.RTPColor,
                )
            except BaseException:
                self.destroy()
                raise
        finally:
            core.unloadPrcFile(prc)
        self._recording_ready = True

    def _clear_scene(self):
        for obj in self.objects.values():
            obj.remove(obj.task)
        for light in self.lightsNP.values():
            self.render.clearLight(light)
            light.removeNode()
        if self.present_movie:
            self.mov_texture.stop()
            self.movie_node.removeNode()
            if self.is_recording:
                # A cached MovieTexture retains its decoder cursor and last frame.
                core.TexturePool.releaseTexture(self.mov_texture)
            self.mov_texture = None
            self.present_movie = False

    def _open_capture(self):
        if not self._recording_prepared or self._recording_started:
            raise RuntimeError("Prepare a recording period before starting it")
        # Upstream Agent's timer starts before its delay. The delay gates visibility,
        # and trajectories and the cutoff still use time since period onset.
        duration = self._recording_duration_ms
        if duration is None:
            duration = max(obj.duration for obj in self.objects.values())
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("Recording period duration must be finite and nonnegative")
        self._frame_count = math.ceil(duration * self.fps / 1000)
        self._frame_index = 0
        self._capture_complete = False
        key = self.logger.trial_key
        period = quote(self.period, safe="")
        filename_template = f"trial-{{}}-{{}}.{self.record_format}"
        session_dir = Path("Recordings") / f"{key['animal_id']}_{key['session']}"
        source_dir = self.recording_path
        if source_dir is None:
            source_dir = Path(self.logger.source_path) / session_dir
        else:
            filename_template = f"{key['animal_id']}_{key['session']}_{filename_template}"
        self._recording_template = filename_template
        self.capture_path = source_dir / filename_template.format(key["trial_idx"], period)
        self.capture_path.parent.mkdir(parents=True, exist_ok=True)
        self._recording_target = None
        if self.logger.target_path:
            target_root = Path(self.logger.target_path).resolve()
            # Ethopy defaults target_path to the filesystem root when no target is configured.
            if target_root.parent != target_root and target_root.is_dir():
                self._recording_target = target_root / session_dir
        if self.capture_path.exists():
            raise FileExistsError(self.capture_path)
        if self.record_format != "npy" and self._frame_count == 0:
            raise ValueError(f"{self.record_format.upper()} capture requires at least one frame")

        self._capture_dir = tempfile.TemporaryDirectory(dir=self.capture_path.parent, prefix=".capture-")
        self._pending_capture = Path(self._capture_dir.name) / self.capture_path.name
        width, height = self.win.getXSize(), self.win.getYSize()
        if self.record_format == "png":
            self._pending_capture.mkdir()
        elif self.record_format == "npy":
            self._capture_file = self._pending_capture.open("wb")
            np.lib.format.write_array_header_2_0(
                self._capture_file,
                {"descr": "|u1", "fortran_order": False,
                 "shape": (self._frame_count, height, width, 3)},
            )
        else:
            # 4:2:0 requires even dimensions. Use 4:4:4 to retain odd dimensions.
            pixel_format = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
            try:
                self._encoder = subprocess.Popen(
                    [
                        "ffmpeg", "-loglevel", "error", "-nostdin", "-f", "rawvideo",
                        "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                        "-framerate", str(self.fps), "-i", "pipe:0", "-an",
                        "-c:v", "libx264", "-pix_fmt", pixel_format,
                        str(self._pending_capture),
                    ],
                    stdin=subprocess.PIPE,
                )
            except FileNotFoundError as error:
                raise RuntimeError("Panda MOV capture requires ffmpeg on PATH") from error
            self._capture_file = self._encoder.stdin

    def _capture_frame(self):
        # Render exactly one frame, without taskMgr, real-time clocks, or sleep.
        t = self._frame_index / self.fps
        for obj in self.objects.values():
            obj.update_for_recording(t)
        if self.present_movie:
            self.mov_texture.setTime(t)
        self.graphicsEngine.renderFrame()

        texture = self._capture_texture
        if self.record_format == "png":
            image = core.PNMImage()
            file = self._pending_capture / f"frame-{self._frame_index:06d}.png"
            if not texture.store(image) or not image.write(core.Filename.fromOsSpecific(str(file))):
                raise RuntimeError(f"Panda3D could not write PNG frame {file.name}")
        else:
            width, height = self.win.getXSize(), self.win.getYSize()
            rgb = np.frombuffer(texture.getRamImageAs("RGB"), dtype=np.uint8)
            if rgb.size != width * height * 3:
                raise RuntimeError("Panda3D did not return a full RGB frame")
            frame = np.ascontiguousarray(rgb.reshape(height, width, 3)[::-1])
            self._capture_file.write(memoryview(frame))
        self._frame_index += 1
        self._capture_complete = self._frame_index == self._frame_count
        self.in_operation = not self._capture_complete

    def _finish_capture(self):
        if self._capture_dir is None:
            return
        try:
            try:
                if self._capture_file is not None:
                    self._capture_file.close()
            finally:
                if self._encoder is not None:
                    status = self._encoder.wait()
                    if self._capture_complete and status:
                        raise RuntimeError(f"ffmpeg exited with status {status}")
            if self._capture_complete:
                if self.record_format == "png":
                    if self.capture_path.exists():
                        raise FileExistsError(self.capture_path)
                    self._pending_capture.rename(self.capture_path)
                else:
                    os.link(self._pending_capture, self.capture_path)
                self._copy_and_log_recording()
        finally:
            self._capture_dir.cleanup()
            self._capture_dir = self._capture_file = self._encoder = None

    def _copy_and_log_recording(self):
        target_dir = self._recording_target
        if target_dir is not None:
            target = target_dir / self.capture_path.name
            if target.resolve() != self.capture_path.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(target)
                with tempfile.TemporaryDirectory(dir=target.parent) as temp_dir:
                    staged = Path(temp_dir) / target.name
                    if self.capture_path.is_dir():
                        shutil.copytree(self.capture_path, staged)
                    else:
                        shutil.copy2(self.capture_path, staged)
                    staged.rename(target)

        key = self.logger.trial_key
        session = (key["animal_id"], key["session"])
        if self.recording_db_log and self._logged_recording_session != session:
            self.logger.log_recording(
                dict(
                    rec_aim=self.recording_aim,
                    software="EthoPy",
                    version=__version__,
                    filename=self._recording_template,
                    source_path=os.path.join(str(self.capture_path.parent), ""),
                    target_path=os.path.join(str(target_dir), "") if target_dir else "",
                ),
                block=True,
            )
            self._logged_recording_session = session

    def _stop_recording(self):
        if not self._recording_prepared:
            return
        self.in_operation = False
        try:
            self._finish_capture()
        finally:
            self._clear_scene()
            self._recording_prepared = False
            if self._recording_started:
                self._recording_started = False
                self.log_stop()

    def fill(self, color=None):
        if self.is_recording:
            # Blank/ready fills do not belong to a captured stimulus period.
            return
        if not color:
            color = self.curr_cond["background_color"]
        self.set_background_color(*color)
        self.flip(2)

    def close(self):
        pass

    def exit(self):
        if self.is_recording:
            if self._recording_ready:
                try:
                    self._stop_recording()
                finally:
                    self._recording_ready = False
                    self.destroy()
            return
        self.destroy()

    def get_cond(self, cond_name, idx=0):
        return {
            key[len(cond_name):]: (
                value if idx is None or np.isscalar(value) or value is None
                else value[idx]
            )
            for key, value in self.curr_cond.items()
            if key.startswith(cond_name)
        }

    def make_conditions(self, conditions):
        conditions = self._log_panda_conditions(conditions)

        # Keep local copies of the stimulus assets.
        os.makedirs(self.path, exist_ok=True)
        for cond in conditions:
            if "movie_name" in cond:
                file_name = self.get_clip_info(cond, "file_name")[0]
                filename = self.movie_path + file_name
                if not os.path.isfile(filename):
                    print("Saving %s" % filename)
                    self.get_clip_info(cond, "clip")[0].tofile(filename)
            if "obj_id" not in cond:
                continue
            for obj_id in iterable(cond["obj_id"]):
                object_info = self.exp.logger.get(
                    schema="stimulus", table="Objects", key={"obj_id": obj_id}
                )[0]
                filename = self.path + object_info["file_name"]
                self.object_files[obj_id] = filename
                if not os.path.isfile(filename):
                    print("Saving %s" % filename)
                    object_info["object"].tofile(filename)
        return conditions

    def _log_panda_conditions(self, conditions):
        movie_fields = {"movie_name", "clip_number"}
        for condition in conditions:
            supplied = movie_fields.intersection(self.default_key.keys() | condition.keys())
            if supplied and supplied != movie_fields:
                raise ValueError("Panda movie conditions require movie_name and clip_number")

        tables = self.cond_tables
        object_tables = [table for table in tables if table != "Panda.Movie"]
        logged = []
        try:
            # Keep the original order, including batches that mix the two types.
            for has_movie, group in groupby(
                conditions,
                key=lambda cond: "movie_name" in cond or "movie_name" in self.default_key,
            ):
                self.cond_tables = tables if has_movie else object_tables
                logged.extend(super().make_conditions(list(group)))
        finally:
            self.cond_tables = tables
        return logged

    def get_clip_info(self, key, *fields):
        return self.logger.get(
            schema="stimulus", table="Movie.Clip", key=key, fields=fields
        )

    def interp(self, points, n_samples=100):
        points = np.atleast_1d(np.asarray(points, dtype=float))

        if points.size == 1:
            return np.full(n_samples, points[0])

        knot_times = np.linspace(0.0, 1.0, points.size)
        sample_times = np.linspace(0.0, 1.0, n_samples)

        return interpolate.PchipInterpolator(knot_times, points)(sample_times)

    def set_taskMgr(self):
        """
        Use this at the setup of pandas because for some reason the taskMgr the first time it
        doesn't work properly. It needs time sleep between steps or to run many steps
        """
        self.set_background_color((0, 0, 0))
        for i in range(0, 2):
            time.sleep(0.1)
            self.taskMgr.step()


class Agent(Panda):
    def __init__(self, env, cond):
        self.cond = cond
        self.env = env
        self.timer = Timer()
        self.duration = cond["dur"]
        hfov = self.env.camLens.get_fov() * np.pi / 180  # half field of view in radians
        # define object time parameters
        self.rot_fun = self.time_fun(cond["rot"])
        self.tilt_fun = self.time_fun(cond["tilt"])
        self.yaw_fun = self.time_fun(cond["yaw"])
        self.z_loc = 2
        if cond["occluder"]:
            self.z_loc = 1.5
        # Transform the knots once, then interpolate them for each frame.
        x_positions = np.arctan(np.asarray(cond["pos_x"]) * hfov[0]) * self.z_loc
        y_positions = np.arctan(np.asarray(cond["pos_y"]) * hfov[0]) * self.z_loc
        scales = 0.15 * np.asarray(cond["mag"])
        self.x_fun = self.time_fun(x_positions)
        self.y_fun = self.time_fun(y_positions)
        self.scale_fun = self.time_fun(scales)
        self.animation_fun = None
        if cond.get("animation") is not None:
            progress = np.atleast_1d(np.asarray(cond["animation"], dtype=float))
            if progress.ndim != 1 or not progress.size or not np.isfinite(progress).all():
                raise ValueError("animation must be a finite scalar or a nonempty 1-D sequence")
            if progress.size == 1:
                # Use a constant trajectory so fixed poses follow the same update path.
                progress = np.repeat(progress, 2)
            self.animation_fun = self.time_fun(progress)
        # add task object
        self.name = "Obj%s-Task" % cond["id"]
        self.model = None
        self.task = None

    def run(self):
        self.timer.start()
        self.load_model()
        self.task = self.env.taskMgr.doMethodLater(
            self.cond["delay"] / 1000, self.objTask, self.name
        )

    def load_model(self):
        self.model = self.env.loader.loadModel(self.env.object_files[self.cond["id"]])
        self.animation_name = None
        if self.animation_fun is not None and (
            isinstance(self.model.node(), core.Character)
            or not self.model.find("**/+Character").isEmpty()
        ):
            actor = Actor(self.model)
            clips = actor.getAnimNames()
            if clips:
                clip = clips[0]
                frame_count = actor.getNumFrames(clip)
                if frame_count:
                    self.model = actor
                    self.animation_name = clip
                    self.last_animation_frame = frame_count - 1
                    self.pose_animation(self.animation_fun(0))
            if self.animation_name is None:
                actor.cleanup()

        self.model.reparentTo(self.env.render)
        if self.cond["occluder"]:
            self.model.setLightOff()
            self.model.setColor(*self.env.curr_cond["background_color"])

    def pose_animation(self, progress):
        progress = float(np.clip(progress, 0.0, 1.0))
        frame = round(progress * self.last_animation_frame)
        self.model.pose(self.animation_name, frame)

    def objTask(self, task):
        t = self.timer.elapsed_time() / 1000
        if t > self.duration / 1000:
            self.remove(task)
            return
        self.update_pose(t)
        return Task.cont

    def update_for_recording(self, t):
        if t < self.cond["delay"] / 1000 or t > self.duration / 1000:
            self.model.hide()
            return
        self.model.show()
        self.update_pose(t)

    def update_pose(self, t):
        self.model.setHpr(self.rot_fun(t), self.tilt_fun(t), self.yaw_fun(t))
        self.model.setPos(self.x_fun(t), self.z_loc, self.y_fun(t))
        self.model.setScale(self.scale_fun(t))
        if self.animation_name is not None:
            self.pose_animation(self.animation_fun(t))

    def remove(self, task):
        if self.model is None or self.model.isEmpty():
            return
        if task is not None:
            task.remove()
        if isinstance(self.model, Actor):
            self.model.cleanup()
        else:
            self.model.removeNode()

    def time_fun(self, values):
        values = np.atleast_1d(values)
        times = np.linspace(0, self.duration / 1000, values.size)
        return lambda t: np.interp(t, times, values)
