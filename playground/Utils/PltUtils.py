"""
Time series plotting utility for real-time data visualization.
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import time
from typing import Callable, Optional, Tuple, Any, List, Dict
from collections import deque

class TimeSeriesPlotter:
    """
    Class for visualizing time-varying data in real time.
    Supports multiple data series plotting.
    """
    def __init__(
        self,
        max_points: int = 1000,
        title: str = "data",
        xlabel: str = "time/s",
        ylabel: str = "value",
        y_range: Optional[Tuple[float, float]] = None,
        series_config: Optional[Dict[str, Dict]] = None,
    ) -> None:
        """
        Initialize the time series plotter

        Args:
            max_points: Maximum number of data points
            title: Chart title
            xlabel: x-axis label
            ylabel: y-axis label
            y_range: y-axis range (min, max), None for automatic adjustment
            series_config: Configuration for multiple data series
                          Format: {"series_name": {"color": "r", "linestyle": "-", "label": "label"}}
        """
        self.max_points = max_points
        self.times: deque = deque(maxlen=max_points)
        self.start_time = time.time()
        
        # Initialize data storage for multiple series
        self.series_data: Dict[str, deque] = {}
        self.series_lines: Dict[str, Any] = {}
        
        # Default series configuration
        self.default_colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
        self.default_linestyles = ['-', '--', '-.', ':']
        
        # Setup series configuration
        if series_config is None:
            # Default to single series for backward compatibility
            series_config = {"data": {"color": "b", "linestyle": "-", "label": "data"}}
        
        self.series_config = series_config
        
        plt.ion()
        self.fig, self.ax = plt.subplots()
        
        # Initialize data storage and plot lines for each series
        for i, (series_name, config) in enumerate(series_config.items()):
            self.series_data[series_name] = deque(maxlen=max_points)
            
            # Get color and linestyle
            color = config.get("color", self.default_colors[i % len(self.default_colors)])
            linestyle = config.get("linestyle", self.default_linestyles[i % len(self.default_linestyles)])
            label = config.get("label", series_name)
            
            # Create plot line
            line, = self.ax.plot([], [], color=color, linestyle=linestyle, label=label)
            self.series_lines[series_name] = line
        
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)
        self.ax.legend()
        self.ax.set_xlim(0, 10)
        if y_range:
            self.ax.set_ylim(*y_range)
        else:
            self.ax.set_ylim(0, 1)
        self.fps_text = self.ax.text(0.02, 0.95, "", transform=self.ax.transAxes)
        self.last_time = time.time()
        self.frame_count = 0
        plt.show(block=False)
        plt.pause(0.1)

    def add_data(self, value: float, series_name: str = "data", close_callback: Optional[Callable] = None) -> None:
        """
        Add a new data point for a specific series and update the plot.
        
        Args:
            value: Data value
            series_name: Name of the data series (must be in series_config)
            close_callback: Callback when window is closed
        """
        if series_name not in self.series_data:
            raise ValueError(f"Series '{series_name}' not found in configuration")
        
        current_time = time.time() - self.start_time
        
        # Add time if not already added (all series share the same time axis)
        if len(self.times) == 0 or current_time != self.times[-1]:
            self.times.append(current_time)
        
        # Add value to the specified series
        self.series_data[series_name].append(value)
        
        # Update FPS counter
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = time.time()
            fps = 30 / (now - self.last_time)
            self.fps_text.set_text(f"FPS: {fps:.1f}")
            self.last_time = now
        
        # Update plot lines
        self._update_plot()
        
        # Setup close callback
        if close_callback and not hasattr(self, "_close_callback_set"):
            self.fig.canvas.mpl_connect("close_event", lambda event: close_callback())
            self._close_callback_set = True

    def add_multiple_data(self, data_dict: Dict[str, float], close_callback: Optional[Callable] = None) -> None:
        """
        Add data points for multiple series at once.
        
        Args:
            data_dict: Dictionary mapping series names to values
            close_callback: Callback when window is closed
        """
        current_time = time.time() - self.start_time
        
        # Add time if not already added
        if len(self.times) == 0 or current_time != self.times[-1]:
            self.times.append(current_time)
        
        # Add values to all specified series
        for series_name, value in data_dict.items():
            if series_name in self.series_data:
                self.series_data[series_name].append(value)
            else:
                raise ValueError(f"Series '{series_name}' not found in configuration")
        
        # Update FPS counter
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            now = time.time()
            fps = 30 / (now - self.last_time)
            self.fps_text.set_text(f"FPS: {fps:.1f}")
            self.last_time = now
        
        # Update plot lines
        self._update_plot()
        
        # Setup close callback
        if close_callback and not hasattr(self, "_close_callback_set"):
            self.fig.canvas.mpl_connect("close_event", lambda event: close_callback())
            self._close_callback_set = True

    def _update_plot(self) -> None:
        """Update all plot lines with current data."""
        times_list = list(self.times)
        
        # Update each series line
        for series_name, line in self.series_lines.items():
            if series_name in self.series_data:
                values_list = list(self.series_data[series_name])
                # Ensure we have the same number of points for time and values
                min_len = min(len(times_list), len(values_list))
                if min_len > 0:
                    line.set_data(times_list[:min_len], values_list[:min_len])
        
        # Update axis limits
        if len(times_list) > 0:
            xmin, xmax = min(times_list), max(times_list)
            
            # Calculate y-axis limits from all series
            all_values = []
            for series_data in self.series_data.values():
                if len(series_data) > 0:
                    all_values.extend(list(series_data))
            
            if all_values:
                ymin, ymax = min(all_values), max(all_values)
                margin = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
                self.ax.set_xlim(xmin, xmax + 1)
                self.ax.set_ylim(ymin - margin, ymax + margin)
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def close(self) -> None:
        """
        Close the plot window.
        """
        plt.close(self.fig)
