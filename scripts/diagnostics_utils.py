#!/usr/bin/env python3
"""
Diagnostics utilities for Phase 1.5 square dish simulations.

Functions for:
- Finding Gor'kov potential minima on a plane
- Computing convergence metrics between different mesh resolutions
- Generating diagnostic reports

Author: Acousto-Tweezers Project  
Date: February 2026
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import minimum_filter
from typing import Tuple, List, Dict
import json


def find_gorkov_minima_2d(
    U_grid: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    n_minima: int = 10,
    min_distance_pts: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find local minima in a 2D Gor'kov potential field.
    
    Parameters
    ----------
    U_grid : np.ndarray
        2D array of Gor'kov potential values, shape (ny, nx)
    x_coords : np.ndarray
        1D array of x coordinates, shape (nx,)
    y_coords : np.ndarray
        1D array of y coordinates, shape (ny,)
    n_minima : int
        Maximum number of minima to return
    min_distance_pts : int
        Minimum distance in grid points between minima
        
    Returns
    -------
    positions : np.ndarray
        Shape (n, 2) array of (x, y) positions of minima [m]
    values : np.ndarray
        Shape (n,) array of Gor'kov potential values at minima [J]
    """
    # Find local minima using minimum filter
    # A point is a local minimum if it equals the minimum in its neighborhood
    neighborhood_size = 2 * min_distance_pts + 1
    local_min = minimum_filter(U_grid, size=neighborhood_size)
    minima_mask = (U_grid == local_min)
    
    # Remove boundary pixels to avoid edge effects
    minima_mask[0, :] = False
    minima_mask[-1, :] = False
    minima_mask[:, 0] = False
    minima_mask[:, -1] = False
    
    # Get coordinates of minima
    iy_minima, ix_minima = np.where(minima_mask)
    
    if len(iy_minima) == 0:
        return np.array([]), np.array([])
    
    # Get values at minima
    values_at_minima = U_grid[iy_minima, ix_minima]
    
    # Sort by value (deepest first)
    sort_idx = np.argsort(values_at_minima)
    iy_minima = iy_minima[sort_idx]
    ix_minima = ix_minima[sort_idx]
    values_at_minima = values_at_minima[sort_idx]
    
    # Take only the n_minima deepest
    n_found = min(n_minima, len(values_at_minima))
    iy_minima = iy_minima[:n_found]
    ix_minima = ix_minima[:n_found]
    values_at_minima = values_at_minima[:n_found]
    
    # Convert grid indices to physical coordinates
    positions = np.zeros((n_found, 2))
    positions[:, 0] = x_coords[ix_minima]
    positions[:, 1] = y_coords[iy_minima]
    
    return positions, values_at_minima


def match_minima_nearest_neighbor(
    positions1: np.ndarray,
    positions2: np.ndarray,
    max_distance: float = np.inf
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Match minima between two sets using nearest-neighbor matching.
    
    Parameters
    ----------
    positions1 : np.ndarray
        Shape (n1, 2) array of (x, y) positions from first set
    positions2 : np.ndarray
        Shape (n2, 2) array of (x, y) positions from second set
    max_distance : float
        Maximum distance for a valid match [m]
        
    Returns
    -------
    idx1 : np.ndarray
        Indices into positions1 of matched points
    idx2 : np.ndarray
        Indices into positions2 of matched points
    """
    if len(positions1) == 0 or len(positions2) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    
    n1 = len(positions1)
    n2 = len(positions2)
    
    # Compute distance matrix
    # Broadcasting: positions1[i, :] - positions2[j, :]
    diff = positions1[:, np.newaxis, :] - positions2[np.newaxis, :, :]  # (n1, n2, 2)
    distances = np.linalg.norm(diff, axis=2)  # (n1, n2)
    
    # Greedy nearest-neighbor matching
    matched_idx1 = []
    matched_idx2 = []
    used_j = set()
    
    for i in range(n1):
        # Find nearest unmatched point in set 2
        min_dist = np.inf
        best_j = -1
        for j in range(n2):
            if j not in used_j and distances[i, j] < min_dist:
                min_dist = distances[i, j]
                best_j = j
        
        if best_j >= 0 and min_dist < max_distance:
            matched_idx1.append(i)
            matched_idx2.append(best_j)
            used_j.add(best_j)
    
    return np.array(matched_idx1, dtype=int), np.array(matched_idx2, dtype=int)


def compute_convergence_metrics(
    minima_positions_list: List[np.ndarray],
    minima_values_list: List[np.ndarray],
    k_deepest: int = 5
) -> Dict:
    """
    Compute convergence metrics between multiple mesh resolutions.
    
    Parameters
    ----------
    minima_positions_list : list of np.ndarray
        List of (n, 2) arrays of minima positions for each resolution
    minima_values_list : list of np.ndarray
        List of (n,) arrays of minima values for each resolution
    k_deepest : int
        Number of deepest minima to use for matching
        
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - 'displacements': list of displacement arrays between consecutive resolutions
        - 'mean_displacement': list of mean displacements [m]
        - 'max_displacement': list of max displacements [m]
        - 'n_matched': list of number of matched minima
    """
    n_resolutions = len(minima_positions_list)
    
    if n_resolutions < 2:
        return {
            'displacements': [],
            'mean_displacement': [],
            'max_displacement': [],
            'n_matched': []
        }
    
    displacements_list = []
    mean_displacements = []
    max_displacements = []
    n_matched_list = []
    
    for i in range(n_resolutions - 1):
        pos1 = minima_positions_list[i][:k_deepest]
        pos2 = minima_positions_list[i+1][:k_deepest]
        
        # Match minima
        # Use wavelength/2 as max distance for matching (very generous)
        max_dist = 0.001  # 1 mm (half the domain size)
        idx1, idx2 = match_minima_nearest_neighbor(pos1, pos2, max_distance=max_dist)
        
        if len(idx1) > 0:
            # Compute displacements
            matched_pos1 = pos1[idx1]
            matched_pos2 = pos2[idx2]
            displacements = np.linalg.norm(matched_pos1 - matched_pos2, axis=1)
            
            displacements_list.append(displacements)
            mean_displacements.append(float(np.mean(displacements)))
            max_displacements.append(float(np.max(displacements)))
            n_matched_list.append(len(idx1))
        else:
            displacements_list.append(np.array([]))
            mean_displacements.append(np.nan)
            max_displacements.append(np.nan)
            n_matched_list.append(0)
    
    return {
        'displacements': [d.tolist() for d in displacements_list],
        'mean_displacement': mean_displacements,
        'max_displacement': max_displacements,
        'n_matched': n_matched_list
    }


def save_minima_data(
    minima_positions: np.ndarray,
    minima_values: np.ndarray,
    filepath: str,
    metadata: dict = None
):
    """
    Save minima data to JSON file.
    
    Parameters
    ----------
    minima_positions : np.ndarray
        Shape (n, 2) array of (x, y) positions
    minima_values : np.ndarray
        Shape (n,) array of Gor'kov values
    filepath : str
        Output JSON file path
    metadata : dict, optional
        Additional metadata to include
    """
    data = {
        'n_minima': len(minima_values),
        'minima': []
    }
    
    for i in range(len(minima_values)):
        data['minima'].append({
            'index': i,
            'x': float(minima_positions[i, 0]),
            'y': float(minima_positions[i, 1]),
            'U': float(minima_values[i])
        })
    
    if metadata:
        data['metadata'] = metadata
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_minima_data(filepath: str) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Load minima data from JSON file.
    
    Returns
    -------
    positions : np.ndarray
        Shape (n, 2)
    values : np.ndarray
        Shape (n,)
    metadata : dict
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    n = data['n_minima']
    positions = np.zeros((n, 2))
    values = np.zeros(n)
    
    for item in data['minima']:
        i = item['index']
        positions[i, 0] = item['x']
        positions[i, 1] = item['y']
        values[i] = item['U']
    
    metadata = data.get('metadata', {})
    
    return positions, values, metadata
