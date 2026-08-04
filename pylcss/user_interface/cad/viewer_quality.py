# Copyright (c) 2026 Kutay Demir.
# Licensed under the PolyForm Shield License 1.0.0. See LICENSE.
"""RenderQualityMixin — presentation quality for the CAD/FEA viewer.

Everything here is display-only: no filter in this module changes geometry,
field values, or anything a result depends on.

Three VTK 9.3 constraints shape the design, all of them measured rather than
assumed (a custom pass silently *disables* the alternatives, it does not warn):

* Installing a render pass via ``vtkRenderer::SetPass`` bypasses both MSAA and
  ``UseFXAA``.  With an SSAO pass active, the two normal anti-aliasing routes
  produce byte-identical output to no anti-aliasing at all, so the pass chain
  has to supply its own — hence ``vtkSSAAPass``.
* ``vtkSSAAPass`` must sit *outside* ``vtkSSAOPass``.  The other nesting order
  renders a nearly blank frame.
* Depth peeling does not engage while a pass chain is installed
  (``GetLastRenderingUsedDepthPeeling()`` stays 0).  Correct transparency and
  ambient occlusion are therefore mutually exclusive, which is why
  :meth:`_refresh_render_quality` picks one path per frame based on whether
  the scene actually contains translucent actors.
"""

from __future__ import annotations

import logging

import numpy as np
import vtk
from PySide6 import QtCore

__all__ = ["RenderQualityMixin"]

_LOG = logging.getLogger(__name__)

# Above this cell count the supersampling pass costs more than it returns —
# a 4x fragment load on a multi-million-element mesh stalls interaction.
_SSAA_MAX_CELLS = 1_500_000
# Ambient occlusion samples a hemisphere of this fraction of the model
# diagonal. Too small and the effect vanishes; too large and flat faces bleed.
_SSAO_RADIUS_FRACTION = 0.025
_SSAO_BIAS_FRACTION = 0.001
# Base colour for CAD geometry that does not ask for one of its own.
_CAD_COLOR = (0.40, 0.51, 0.58)


class RenderQualityMixin:
    """Lighting, anti-aliasing, ambient occlusion, materials and backgrounds."""

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------
    def _init_render_quality(self):
        """Install lighting and the pass chain. Safe to call once from __init__."""
        self._quality_enabled = True
        # Physically based shading is the intended look, but it needs a vetted
        # shader context, so the Phong fallback holds until the probe passes —
        # see _probe_gl_capability and _enable_realistic_shading.
        self._shading_mode = "standard"
        self._ssao_pass = None
        self._ssaa_pass = None
        self._pass_chain_installed = False
        self._interacting = False
        self._scene_diagonal = 1.0
        self._env_texture = None
        self._light_kit = None
        self._silhouette_actors = []
        # Nothing that needs a shader may be switched on until a real context
        # has been confirmed — see _probe_gl_capability.
        self._gl_checked = False
        self._gl_capable = False

        self._build_light_kit()
        self._apply_viewport_background(self._light_mode)
        self._build_pass_chain()

        # Quality degrades during camera motion and is restored when the user
        # lets go — the same trick every commercial viewer uses to keep orbit
        # responsive on large models.
        try:
            self.interactor.AddObserver(
                "StartInteractionEvent", self._on_interaction_start
            )
            self.interactor.AddObserver(
                "EndInteractionEvent", self._on_interaction_end
            )
        except Exception:
            _LOG.debug("Could not install interaction-quality observers.", exc_info=True)

        self._refresh_render_quality()

    def _build_light_kit(self):
        """Replace VTK's single headlight with a three-point studio rig.

        A headlight sits at the camera, so every surface facing the viewer is
        lit identically and curvature reads as flat.  The key/fill/back split
        is what makes a machined part look solid; the warm key against a cool
        fill is the convention CAD packages use.
        """
        try:
            kit = vtk.vtkLightKit()
            kit.SetKeyLightIntensity(0.90)
            kit.SetKeyLightWarmth(0.58)
            kit.SetFillLightWarmth(0.42)
            kit.SetHeadLightWarmth(0.50)
            kit.SetBackLightWarmth(0.52)
            kit.SetKeyToFillRatio(2.6)
            kit.SetKeyToHeadRatio(3.5)
            kit.SetKeyToBackRatio(3.0)
            kit.SetKeyLightElevation(50)
            kit.SetKeyLightAzimuth(20)
            kit.SetMaintainLuminance(True)
            kit.AddLightsToRenderer(self.renderer)
            # The overlay layer carries supports and load arrows. Without the
            # same rig they shade differently from the part underneath.
            kit.AddLightsToRenderer(self.bc_renderer)
            self._light_kit = kit
        except Exception:
            _LOG.debug("Could not install the light kit.", exc_info=True)

    def _build_pass_chain(self):
        """Create the SSAA(SSAO(steps)) chain. Installed by _refresh_render_quality."""
        try:
            steps = vtk.vtkRenderStepsPass()
            ssao = vtk.vtkSSAOPass()
            ssao.SetKernelSize(64)
            ssao.SetBlur(True)
            ssao.SetDelegatePass(steps)
            ssaa = vtk.vtkSSAAPass()
            ssaa.SetDelegatePass(ssao)
            self._ssao_pass = ssao
            self._ssaa_pass = ssaa
        except Exception:
            _LOG.debug("Ambient-occlusion passes unavailable.", exc_info=True)
            self._ssao_pass = None
            self._ssaa_pass = None

    # ------------------------------------------------------------------
    # graphics-context capability
    # ------------------------------------------------------------------
    def _probe_gl_capability(self):
        """Decide once whether this context can run the shader-based features.

        VTK 9.3 draws a gradient background through a shader quad and does not
        degrade when the shader will not compile — it dereferences the null
        program and takes the process down.  Contexts that behave this way do
        exist in the field: RDP sessions, VMs without 3-D acceleration, and
        stale drivers all produce one.  ``ReportCapabilities`` names the GL
        version on a working context and returns "no device context" on a
        broken one, which is the cheapest reliable discriminator.

        Must run after a real render: the report is meaningless before the
        context is realised, and probing early would latch a false negative.
        """
        if self._gl_checked:
            return self._gl_capable
        try:
            render_window = self.vtkWidget.GetRenderWindow()
            if not render_window.SupportsOpenGL():
                capable = False
            else:
                report = (render_window.ReportCapabilities() or "").lower()
                capable = "opengl version string" in report
        except Exception:
            _LOG.debug("Could not query graphics capabilities.", exc_info=True)
            capable = False

        self._gl_checked = True
        self._gl_capable = capable
        if not capable:
            _LOG.warning(
                "Graphics context reports no usable OpenGL; the viewer will "
                "run with a flat background and no ambient occlusion."
            )
        return capable

    def _on_first_render(self):
        """Upgrade presentation quality once the context is known good.

        Deferred through the event loop rather than applied inline: this runs
        from the render window's EndEvent, and re-rendering from inside that
        callback re-enters the renderer.
        """
        if self._gl_checked:
            return
        if not self._probe_gl_capability():
            return
        QtCore.QTimer.singleShot(0, self._apply_deferred_quality)

    def _apply_deferred_quality(self):
        try:
            self._apply_viewport_background(self._light_mode)
            self._enable_realistic_shading()
            self._refresh_render_quality()
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            _LOG.debug("Could not upgrade render quality.", exc_info=True)

    # ------------------------------------------------------------------
    # quality path selection
    # ------------------------------------------------------------------
    def _scene_has_translucency(self):
        """True when any visible actor is drawn with alpha < 1."""
        for renderer in (self.renderer, self.bc_renderer):
            try:
                actors = renderer.GetActors()
            except Exception:
                continue
            actors.InitTraversal()
            for _ in range(actors.GetNumberOfItems()):
                actor = actors.GetNextActor()
                if actor is None or not actor.GetVisibility():
                    continue
                try:
                    if actor.GetProperty().GetOpacity() < 0.999:
                        return True
                except Exception:
                    continue
        return False

    def _visible_cell_count(self):
        total = 0
        try:
            actors = self.renderer.GetActors()
        except Exception:
            return 0
        actors.InitTraversal()
        for _ in range(actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            if actor is None or not actor.GetVisibility():
                continue
            get_mapper = getattr(actor, "GetMapper", None)
            if not callable(get_mapper):
                continue
            mapper = get_mapper()
            if mapper is None:
                continue
            try:
                dataset = mapper.GetInput()
                if dataset is not None:
                    total += int(dataset.GetNumberOfCells())
            except Exception:
                continue
        return total

    def _refresh_render_quality(self):
        """Choose the ambient-occlusion path or the transparency path.

        See the module docstring: in VTK 9.3 these cannot be combined, so the
        scene decides.  Opaque geometry — the normal CAD and result case — gets
        occlusion and supersampling; anything translucent gets depth peeling
        with MSAA/FXAA instead, because a wrong-order transparent overlay is a
        worse defect than a missing contact shadow.
        """
        render_window = self.vtkWidget.GetRenderWindow()
        want_ssao = (
            self._quality_enabled
            and self._gl_capable
            and not self._interacting
            and self._ssaa_pass is not None
            and not self._scene_has_translucency()
        )

        if want_ssao:
            self._update_scene_scale()
            cells = self._visible_cell_count()
            # Drop supersampling on heavy meshes but keep the occlusion, which
            # is the part that actually conveys shape.
            head = self._ssaa_pass if cells <= _SSAA_MAX_CELLS else self._ssao_pass
            try:
                self.renderer.SetUseDepthPeeling(False)
                self.renderer.UseFXAAOff()
                self.renderer.SetPass(head)
                self._pass_chain_installed = True
            except Exception:
                _LOG.debug("Could not install the quality pass chain.", exc_info=True)
                self._install_transparency_path(render_window)
        else:
            self._install_transparency_path(render_window)

    def _install_transparency_path(self, render_window):
        """MSAA + FXAA + depth peeling, with no custom pass installed."""
        try:
            self.renderer.SetPass(None)
            self._pass_chain_installed = False
        except Exception:
            _LOG.debug("Could not clear the render pass.", exc_info=True)
        try:
            render_window.SetAlphaBitPlanes(1)
            render_window.SetMultiSamples(8)
            self.renderer.UseFXAAOn()
            self.renderer.SetUseDepthPeeling(True)
            self.renderer.SetMaximumNumberOfPeels(8)
            self.renderer.SetOcclusionRatio(0.0)
        except Exception:
            _LOG.debug("Could not configure the transparency path.", exc_info=True)

    def _on_interaction_start(self, *_args):
        """Drop to the cheap path while the camera is moving."""
        if not self._interacting:
            self._interacting = True
            self._refresh_render_quality()

    def _on_interaction_end(self, *_args):
        """Restore full quality once the camera settles."""
        if self._interacting:
            self._interacting = False
            self._refresh_render_quality()
            try:
                self.vtkWidget.GetRenderWindow().Render()
            except Exception:
                _LOG.debug("Optional UI operation failed.", exc_info=True)

    def _update_scene_scale(self):
        """Re-scale ambient occlusion to the model.

        ``vtkSSAOPass`` takes its radius in world units, so a value tuned on a
        500 mm bracket does nothing on a 5 mm one and swamps a 5 m assembly.
        """
        try:
            bounds = self.renderer.ComputeVisiblePropBounds()
        except Exception:
            return
        if not bounds or len(bounds) != 6 or bounds[0] > bounds[1]:
            return
        diagonal = float(
            np.linalg.norm(
                [
                    bounds[1] - bounds[0],
                    bounds[3] - bounds[2],
                    bounds[5] - bounds[4],
                ]
            )
        )
        if not np.isfinite(diagonal) or diagonal <= 0.0:
            return
        self._scene_diagonal = diagonal
        if self._ssao_pass is not None:
            try:
                self._ssao_pass.SetRadius(_SSAO_RADIUS_FRACTION * diagonal)
                self._ssao_pass.SetBias(_SSAO_BIAS_FRACTION * diagonal)
            except Exception:
                _LOG.debug("Could not rescale ambient occlusion.", exc_info=True)

    # ------------------------------------------------------------------
    # background
    # ------------------------------------------------------------------
    def _apply_viewport_background(self, light):
        """Gradient backdrop for the active theme, flat where unsupported."""
        if light:
            top, bottom = (0.99, 0.99, 1.00), (0.72, 0.77, 0.85)
        else:
            top, bottom = (0.22, 0.25, 0.30), (0.06, 0.07, 0.09)
        try:
            # VTK names Background as the *bottom* stop and Background2 the top.
            self.renderer.SetBackground(*bottom)
            self.renderer.SetBackground2(*top)
        except Exception:
            _LOG.debug("Could not set the background colours.", exc_info=True)
            return
        if not self._gl_capable:
            # A flat clear needs no shader, so it is the safe state before the
            # context has been vetted and the permanent state when it fails.
            try:
                self.renderer.GradientBackgroundOff()
                self.renderer.SetBackground(*[(a + b) / 2.0 for a, b in zip(top, bottom)])
            except Exception:
                _LOG.debug("Could not set a flat background.", exc_info=True)
            return
        try:
            self.renderer.GradientBackgroundOn()
        except Exception:
            _LOG.debug("Could not enable the gradient background.", exc_info=True)
        try:
            self.renderer.SetGradientMode(
                vtk.vtkViewport.GradientModes.VTK_GRADIENT_RADIAL_VIEWPORT_FARTHEST_CORNER
            )
        except Exception:
            # Radial gradients arrived in VTK 9.3; the vertical default is fine.
            _LOG.debug("Radial gradient unavailable.", exc_info=True)

    # ------------------------------------------------------------------
    # materials
    # ------------------------------------------------------------------
    def _environment_texture(self):
        """Build a procedural equirectangular environment map for PBR.

        Physically based metals reflect their surroundings; with no
        environment they render nearly black.  A three-stop sky/horizon/ground
        ramp is enough to make the shading read correctly and costs no asset.
        """
        if self._env_texture is not None:
            return self._env_texture
        try:
            width, height = 128, 64
            sky = np.array([0.42, 0.52, 0.68])
            horizon = np.array([0.86, 0.88, 0.92])
            ground = np.array([0.24, 0.23, 0.21])
            t = np.linspace(0.0, 1.0, height)[:, None]
            upper = sky + (horizon - sky) * np.clip(t / 0.5, 0, 1)
            lower = horizon + (ground - horizon) * np.clip((t - 0.5) / 0.5, 0, 1)
            ramp = np.where(t < 0.5, upper, lower)
            rgb = np.repeat(ramp[:, None, :], width, axis=1)
            # A soft bright patch stands in for a key light so curved metal
            # picks up a highlight that moves with the camera.
            yy, xx = np.mgrid[0:height, 0:width]
            glow = np.exp(-(((xx - width * 0.3) / 14.0) ** 2
                            + ((yy - height * 0.28) / 9.0) ** 2))
            rgb = np.clip(rgb + glow[..., None] * 0.55, 0.0, 1.0)

            image = vtk.vtkImageData()
            image.SetDimensions(width, height, 1)
            image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 3)
            flat = (rgb[::-1] * 255.0).astype(np.uint8).reshape(-1, 3)
            scalars = image.GetPointData().GetScalars()
            for i, (r, g, b) in enumerate(flat):
                scalars.SetTuple3(i, int(r), int(g), int(b))

            texture = vtk.vtkTexture()
            texture.SetInputData(image)
            texture.MipmapOn()
            texture.InterpolateOn()
            try:
                texture.UseSRGBColorSpaceOn()
            except Exception:
                _LOG.debug("sRGB environment sampling unavailable.", exc_info=True)
            self._env_texture = texture
        except Exception:
            _LOG.debug("Could not build the environment texture.", exc_info=True)
            self._env_texture = None
        return self._env_texture

    def _enable_realistic_shading(self):
        """Adopt physically based metal shading for CAD geometry.

        This is the viewer's one shading style, not an option: it is switched
        on as soon as the context is known good and never switched back off.
        PBR and image-based lighting both need working shader support, so they
        follow the same gate as the gradient background — see
        :meth:`_probe_gl_capability`.  A context that fails that probe keeps
        the Phong fallback in :meth:`apply_cad_material`, which is the only
        reason that branch still exists.

        Runs after the first render, so CAD actors built before the probe
        completed are restyled here rather than left on the fallback.
        """
        if self._shading_mode == "realistic" or not self._gl_capable:
            return
        texture = self._environment_texture()
        if texture is None:
            # Metal with nothing to reflect renders nearly black, which is a
            # worse result than the Phong fallback.
            return
        try:
            self.renderer.UseImageBasedLightingOn()
            self.renderer.SetEnvironmentTexture(texture)
        except Exception:
            _LOG.debug("Image-based lighting unavailable.", exc_info=True)
            return

        self._shading_mode = "realistic"
        for actor in [self.current_actor, *getattr(self, "actors", [])]:
            if actor is None or getattr(actor, "_bc_overlay", False):
                continue
            if getattr(actor, "_pylcss_material", None) == "cad":
                self.apply_cad_material(
                    actor, color=getattr(actor, "_pylcss_material_color", _CAD_COLOR)
                )
        try:
            self.vtkWidget.GetRenderWindow().Render()
        except Exception:
            _LOG.debug("Optional UI operation failed.", exc_info=True)

    def apply_cad_material(self, actor, color=_CAD_COLOR):
        """Style an actor as machined metal.

        Normally physically based: metal is defined by its metallic and
        roughness response, and the environment map supplies the reflections
        that make a milled face read as milled.  The Phong fallback below is
        for contexts without usable shaders, where the stock VTK look — broad
        specular at power 20 — would otherwise read as plastic.
        """
        if actor is None:
            return
        try:
            prop = actor.GetProperty()
        except Exception:
            return
        actor._pylcss_material = "cad"
        # Kept so a restyle at the shading switch-over does not silently
        # repaint a part that asked for its own colour.
        actor._pylcss_material_color = tuple(color)
        try:
            if self._shading_mode == "realistic":
                prop.SetInterpolationToPBR()
                prop.SetColor(*color)
                prop.SetMetallic(0.32)
                prop.SetRoughness(0.38)
                # An actor restyled from the fallback still carries Phong
                # coefficients; reset them to VTK's defaults so it matches an
                # actor created once PBR is already active.
                prop.SetAmbient(0.0)
                prop.SetDiffuse(1.0)
                prop.SetSpecular(0.0)
                try:
                    prop.SetCoatStrength(0.0)
                    prop.SetEdgeTint(0.92, 0.93, 0.95)
                except Exception:
                    _LOG.debug("Extended PBR knobs unavailable.", exc_info=True)
            else:
                prop.SetInterpolationToPhong()
                prop.SetColor(*color)
                prop.SetAmbient(0.18)
                prop.SetDiffuse(0.72)
                prop.SetSpecular(0.32)
                prop.SetSpecularPower(48)
        except Exception:
            _LOG.debug("Could not apply the CAD material.", exc_info=True)

    def apply_result_material(self, actor):
        """Style an actor carrying a result field.

        Contour colours have to be read against the legend, so the specular
        term is kept low here — a hot highlight shifts the apparent colour and
        makes the plot lie.  PBR is deliberately never used for result actors
        for the same reason.
        """
        if actor is None:
            return
        try:
            prop = actor.GetProperty()
            actor._pylcss_material = "result"
            prop.SetInterpolationToGouraud()
            prop.SetAmbient(0.28)
            prop.SetDiffuse(0.70)
            prop.SetSpecular(0.10)
            prop.SetSpecularPower(20)
        except Exception:
            _LOG.debug("Could not apply the result material.", exc_info=True)

    # ------------------------------------------------------------------
    # render hook
    # ------------------------------------------------------------------
    def render_simulation(self, data, is_sub_render=False):
        """Re-fit presentation quality after the result renderers have run.

        ``RenderQualityMixin`` precedes ``SimulationRenderingMixin`` in the
        MRO, so this wraps the real implementation rather than replacing it.
        Hooking here — instead of at each of the renderer's many exit points —
        keeps the scale and pass-path decisions in one place.
        """
        result = super().render_simulation(data, is_sub_render=is_sub_render)
        if is_sub_render:
            return result
        try:
            self._apply_result_materials()
            self._update_scene_scale()
            self._refresh_render_quality()
        except Exception:
            _LOG.debug("Could not refresh render quality.", exc_info=True)
        return result

    def _apply_result_materials(self):
        """Damp specular on actors that carry a colour-mapped field.

        A hot highlight shifts the apparent colour of a contour band, so a
        value read off the model stops matching the legend.  Actors without
        scalar colouring keep whatever material their renderer chose.
        """
        for actor in [self.current_actor, *getattr(self, "actors", [])]:
            if actor is None or getattr(actor, "_bc_overlay", False):
                continue
            if getattr(actor, "_pylcss_material", None) is not None:
                continue
            get_mapper = getattr(actor, "GetMapper", None)
            if not callable(get_mapper):
                continue
            mapper = get_mapper()
            if mapper is None:
                continue
            try:
                if not mapper.GetScalarVisibility():
                    continue
                if mapper.GetInput() is None:
                    continue
                if mapper.GetInput().GetPointData().GetScalars() is None:
                    continue
            except Exception:
                continue
            self.apply_result_material(actor)

    # ------------------------------------------------------------------
    # silhouette
    # ------------------------------------------------------------------
    def attach_silhouette(self, actor, color=(0.10, 0.11, 0.13), width=2.0):
        """Draw a view-dependent outline around *actor*.

        The outline is what separates a part from the backdrop in a screenshot;
        it recomputes per frame because it depends on the camera direction.
        """
        if actor is None:
            return None
        try:
            mapper = actor.GetMapper()
            if mapper is None:
                return None
            mapper.Update()
            dataset = mapper.GetInput()
            if dataset is None or dataset.GetNumberOfPolys() == 0:
                return None
            silhouette = vtk.vtkPolyDataSilhouette()
            silhouette.SetInputData(dataset)
            silhouette.SetCamera(self.renderer.GetActiveCamera())
            silhouette.SetEnableFeatureAngle(0)
            sil_mapper = vtk.vtkPolyDataMapper()
            sil_mapper.SetInputConnection(silhouette.GetOutputPort())
            sil_mapper.ScalarVisibilityOff()
            sil_actor = vtk.vtkActor()
            sil_actor.SetMapper(sil_mapper)
            sil_actor.GetProperty().SetColor(*color)
            sil_actor.GetProperty().SetLineWidth(width)
            sil_actor.GetProperty().LightingOff()
            sil_actor.PickableOff()
            sil_actor._pylcss_silhouette = True
            self.renderer.AddActor(sil_actor)
            self._silhouette_actors.append(sil_actor)
            return sil_actor
        except Exception:
            _LOG.debug("Could not build a silhouette.", exc_info=True)
            return None

    def clear_silhouettes(self):
        for actor in list(getattr(self, "_silhouette_actors", [])):
            try:
                self.renderer.RemoveActor(actor)
            except Exception:
                _LOG.debug("Optional UI operation failed.", exc_info=True)
        self._silhouette_actors = []
