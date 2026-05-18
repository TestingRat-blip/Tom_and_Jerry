"""Pygame renderer for Replay objects.

Renders a recorded episode with:
  - Grid on the left (32px tiles): walls, vents, lockers, scent field
  - Tom and Jerry as colored circles (red and blue) with facing arrows
  - Side panel on the right: tick, state, action, reward, controls hint
  - Sound events flash briefly when emitted

Controls:
  SPACE      pause / unpause
  →          step forward one tick (when paused)
  ←          step backward one tick (when paused)
  ↑          increase playback speed
  ↓          decrease playback speed
  R          restart from beginning
  ESC / Q    quit
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pygame

from src.env.world.world import EventType
from src.render.replay.recorder import Frame, Replay
from src.utils.types import Action, TileType


# ---- visual config -----------------------------------------------------

@dataclass(frozen=True, slots=True)
class RenderConfig:
    tile_px: int = 32
    panel_width_px: int = 280
    panel_padding_px: int = 16
    fps_base: int = 8           # baseline playback speed (ticks per second)
    fps_min: int = 1
    fps_max: int = 60

    # Colors (R, G, B)
    bg: tuple = (18, 18, 22)
    grid_line: tuple = (40, 40, 48)
    wall: tuple = (60, 60, 68)
    empty: tuple = (28, 28, 34)
    vent: tuple = (90, 130, 180)
    locker: tuple = (140, 100, 60)
    tom: tuple = (220, 80, 80)
    tom_outline: tuple = (255, 200, 200)
    jerry: tuple = (80, 150, 220)
    jerry_outline: tuple = (200, 220, 255)
    jerry_in_locker: tuple = (60, 90, 130)
    panel_bg: tuple = (24, 24, 30)
    text_primary: tuple = (220, 220, 230)
    text_dim: tuple = (140, 140, 150)
    text_warn: tuple = (220, 180, 80)
    scent_max: tuple = (60, 180, 80)   # green-ish at full strength
    sight_line: tuple = (220, 220, 80)
    sound_flash: tuple = (230, 220, 80)
    catch_flash: tuple = (255, 80, 80)


# ---- the renderer ------------------------------------------------------

class ReplayRenderer:
    """Pygame window that plays back a Replay."""

    def __init__(self, replay: Replay, config: RenderConfig | None = None):
        self.replay = replay
        self.config = config or RenderConfig()

        pygame.init()
        pygame.display.set_caption(
            f"Tom & Jerry — {replay.jerry_policy_label} vs {replay.tom_policy_label}"
        )

        self.grid_px_w = replay.grid_width * self.config.tile_px
        self.grid_px_h = replay.grid_height * self.config.tile_px
        self.win_w = self.grid_px_w + self.config.panel_width_px
        self.win_h = max(self.grid_px_h, 640)

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        self.clock = pygame.time.Clock()
        # Fonts
        self.font_small = pygame.font.SysFont("consolas", 14)
        self.font_med = pygame.font.SysFont("consolas", 18)
        self.font_large = pygame.font.SysFont("consolas", 22, bold=True)

        # Playback state
        self.cur_frame: int = 0
        self.paused: bool = False
        self.fps: int = self.config.fps_base
        self._last_sound_tick: int = -10
        self._catch_flash_remaining: int = 0  # render frames remaining for catch flash

    def run(self) -> None:
        """Main event/render loop. Blocks until the window is closed."""
        running = True
        while running:
            # Events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)

            # Advance playback if not paused
            if not self.paused and self.cur_frame < len(self.replay.frames) - 1:
                self.cur_frame += 1
            elif not self.paused and self.cur_frame >= len(self.replay.frames) - 1:
                # End of replay reached — pause automatically so the
                # final frame stays on screen
                self.paused = True

            # Draw
            self._draw()
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()

    # ---- input ---------------------------------------------------------

    def _handle_key(self, key: int) -> bool:
        """Return False to quit, True to keep running."""
        if key in (pygame.K_ESCAPE, pygame.K_q):
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_RIGHT:
            self.cur_frame = min(self.cur_frame + 1, len(self.replay.frames) - 1)
            self.paused = True
        elif key == pygame.K_LEFT:
            self.cur_frame = max(self.cur_frame - 1, 0)
            self.paused = True
        elif key == pygame.K_UP:
            self.fps = min(self.fps + 2, self.config.fps_max)
        elif key == pygame.K_DOWN:
            self.fps = max(self.fps - 2, self.config.fps_min)
        elif key == pygame.K_r:
            self.cur_frame = 0
            self.paused = False
            self._catch_flash_remaining = 0
        return True

    # ---- drawing -------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill(self.config.bg)
        if not self.replay.frames:
            return
        frame = self.replay.frames[self.cur_frame]
        self._draw_grid(frame)
        self._draw_scent(frame)
        self._draw_agents(frame)
        self._draw_sound_events(frame)
        self._draw_events(frame)
        self._draw_panel(frame)

    def _tile_rect(self, x: int, y: int) -> pygame.Rect:
        tp = self.config.tile_px
        return pygame.Rect(x * tp, y * tp, tp, tp)

    def _draw_grid(self, frame: Frame) -> None:
        cfg = self.config
        for y, row in enumerate(self.replay.grid_tiles):
            for x, tile in enumerate(row):
                rect = self._tile_rect(x, y)
                if tile == TileType.WALL:
                    pygame.draw.rect(self.screen, cfg.wall, rect)
                elif tile == TileType.VENT:
                    pygame.draw.rect(self.screen, cfg.empty, rect)
                    inner = rect.inflate(-8, -8)
                    pygame.draw.rect(self.screen, cfg.vent, inner)
                    # Crosshatch to make vents recognizable
                    pygame.draw.line(self.screen, cfg.empty,
                                     inner.topleft, inner.bottomright, 2)
                    pygame.draw.line(self.screen, cfg.empty,
                                     inner.topright, inner.bottomleft, 2)
                elif tile == TileType.LOCKER:
                    pygame.draw.rect(self.screen, cfg.empty, rect)
                    inner = rect.inflate(-6, -6)
                    pygame.draw.rect(self.screen, cfg.locker, inner)
                    # Door line
                    mid = inner.left + inner.width // 2
                    pygame.draw.line(self.screen, cfg.empty,
                                     (mid, inner.top), (mid, inner.bottom), 2)
                else:
                    pygame.draw.rect(self.screen, cfg.empty, rect)
        # Subtle grid lines
        for x in range(self.replay.grid_width + 1):
            px = x * cfg.tile_px
            pygame.draw.line(self.screen, cfg.grid_line,
                             (px, 0), (px, self.grid_px_h))
        for y in range(self.replay.grid_height + 1):
            py = y * cfg.tile_px
            pygame.draw.line(self.screen, cfg.grid_line,
                             (0, py), (self.grid_px_w, py))

    def _draw_scent(self, frame: Frame) -> None:
        """Render Jerry's scent trail as green-tinted overlays."""
        cfg = self.config
        tp = cfg.tile_px
        for x, y, value in frame.scent_cells:
            alpha = int(min(value, 1.0) * 140)  # cap alpha at 140 for subtlety
            if alpha < 10:
                continue
            surf = pygame.Surface((tp, tp), pygame.SRCALPHA)
            surf.fill((*cfg.scent_max, alpha))
            self.screen.blit(surf, (x * tp, y * tp))

    def _draw_agents(self, frame: Frame) -> None:
        cfg = self.config
        tp = cfg.tile_px
        # Tom
        tx, ty = frame.tom_pos
        tom_center = (tx * tp + tp // 2, ty * tp + tp // 2)
        pygame.draw.circle(self.screen, cfg.tom, tom_center, tp // 3)
        pygame.draw.circle(self.screen, cfg.tom_outline, tom_center, tp // 3, 2)
        self._draw_facing_arrow(tom_center, Action(frame.tom_facing), cfg.tom_outline)

        # Jerry — different color if hiding
        jx, jy = frame.jerry_pos
        jerry_color = cfg.jerry_in_locker if frame.jerry_in_locker else cfg.jerry
        jerry_center = (jx * tp + tp // 2, jy * tp + tp // 2)
        pygame.draw.circle(self.screen, jerry_color, jerry_center, tp // 4)
        if not frame.jerry_in_locker:
            pygame.draw.circle(self.screen, cfg.jerry_outline, jerry_center, tp // 4, 2)
            self._draw_facing_arrow(jerry_center, Action(frame.jerry_facing),
                                    cfg.jerry_outline)

        # Sight line if Tom sees Jerry
        if frame.tom_sees_jerry and not frame.jerry_in_locker:
            pygame.draw.line(self.screen, cfg.sight_line, tom_center, jerry_center, 1)

    def _draw_facing_arrow(self, center, facing: Action, color) -> None:
        """Small triangle indicating facing direction."""
        cx, cy = center
        size = 5
        directions = {
            Action.NORTH: [(cx, cy - size), (cx - size, cy), (cx + size, cy)],
            Action.SOUTH: [(cx, cy + size), (cx - size, cy), (cx + size, cy)],
            Action.EAST: [(cx + size, cy), (cx, cy - size), (cx, cy + size)],
            Action.WEST: [(cx - size, cy), (cx, cy - size), (cx, cy + size)],
        }
        if facing in directions:
            pygame.draw.polygon(self.screen, color, directions[facing])

    def _draw_sound_events(self, frame: Frame) -> None:
        """Briefly flash circles where sound was emitted this tick."""
        cfg = self.config
        tp = cfg.tile_px
        for x, y, intensity in frame.sound_events:
            cx = x * tp + tp // 2
            cy = y * tp + tp // 2
            radius = int(tp * 0.4 * min(intensity / 3.0, 1.0))
            if radius < 2:
                continue
            surf = pygame.Surface((tp * 2, tp * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*cfg.sound_flash, 100), (tp, tp), radius, 2)
            self.screen.blit(surf, (cx - tp, cy - tp))

    def _draw_events(self, frame: Frame) -> None:
        """Catch flash if TOM_CAUGHT_JERRY fired."""
        if any(e == EventType.TOM_CAUGHT_JERRY for e in frame.events):
            self._catch_flash_remaining = 12  # render frames
        if self._catch_flash_remaining > 0:
            alpha = int(60 * (self._catch_flash_remaining / 12))
            overlay = pygame.Surface((self.grid_px_w, self.grid_px_h), pygame.SRCALPHA)
            overlay.fill((*self.config.catch_flash, alpha))
            self.screen.blit(overlay, (0, 0))
            self._catch_flash_remaining -= 1

    def _draw_panel(self, frame: Frame) -> None:
        cfg = self.config
        panel_x = self.grid_px_w
        pygame.draw.rect(self.screen, cfg.panel_bg,
                         (panel_x, 0, cfg.panel_width_px, self.win_h))
        x = panel_x + cfg.panel_padding_px
        y = cfg.panel_padding_px

        def line(text: str, font=self.font_small, color=cfg.text_primary, gap: int = 4):
            nonlocal y
            surf = font.render(text, True, color)
            self.screen.blit(surf, (x, y))
            y += surf.get_height() + gap

        # Title
        line("TOM AND JERRY", self.font_large, gap=8)
        line(f"Jerry: {self.replay.jerry_policy_label}", self.font_small, cfg.text_dim)
        line(f"Tom:   {self.replay.tom_policy_label}", self.font_small, cfg.text_dim,
             gap=12)

        # Outcome banner (when ending reached)
        if self.cur_frame == len(self.replay.frames) - 1:
            outcome_color = cfg.text_warn if self.replay.outcome == "caught" \
                else cfg.scent_max
            line(f"OUTCOME: {self.replay.outcome.upper()}", self.font_med,
                 outcome_color, gap=12)

        # Tick / playback
        line(f"Tick:  {frame.tick} / {self.replay.total_ticks}", self.font_med)
        line(f"Frame: {self.cur_frame + 1} / {len(self.replay.frames)}",
             self.font_small, cfg.text_dim)
        line(f"Speed: {self.fps} fps  {'PAUSED' if self.paused else ''}",
             self.font_small, cfg.text_dim, gap=12)

        # Tom info
        line("--- TOM ---", self.font_small, cfg.text_dim)
        line(f"Pos:    {frame.tom_pos}")
        line(f"State:  {frame.tom_state or '-'}")
        line(f"Action: {Action(frame.tom_action).name}")
        line(f"Sees Jerry: {frame.tom_sees_jerry}",
             color=cfg.text_warn if frame.tom_sees_jerry else cfg.text_primary,
             gap=12)

        # Jerry info
        line("--- JERRY ---", self.font_small, cfg.text_dim)
        line(f"Pos:     {frame.jerry_pos}")
        line(f"Action:  {Action(frame.jerry_action).name}")
        line(f"Locker:  {frame.jerry_in_locker}")
        line(f"Alive:   {frame.jerry_alive}",
             color=cfg.text_warn if not frame.jerry_alive else cfg.text_primary)
        line(f"Reward this tick:  {frame.jerry_reward:+.3f}",
             color=cfg.text_warn if frame.jerry_reward < 0 else cfg.text_primary)
        line(f"Reward cumulative: {frame.jerry_cum_reward:+.2f}",
             color=cfg.text_warn if frame.jerry_cum_reward < 0 else cfg.text_primary,
             gap=16)

        # Controls
        line("--- CONTROLS ---", self.font_small, cfg.text_dim)
        line("SPACE  pause/play", self.font_small, cfg.text_dim)
        line("← →    step backward/forward", self.font_small, cfg.text_dim)
        line("↑ ↓    speed up/down", self.font_small, cfg.text_dim)
        line("R      restart", self.font_small, cfg.text_dim)
        line("ESC    quit", self.font_small, cfg.text_dim)


def watch_replay(replay: Replay | str | Path,
                 config: RenderConfig | None = None) -> None:
    """Convenience: load (if a path) and watch a replay."""
    if isinstance(replay, (str, Path)):
        replay = Replay.load(replay)
    renderer = ReplayRenderer(replay, config=config)
    renderer.run()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.render.pygame_renderer.renderer <replay.json>")
        sys.exit(1)
    watch_replay(sys.argv[1])
