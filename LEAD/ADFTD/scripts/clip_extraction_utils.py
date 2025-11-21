import numpy as np
from typing import List, Tuple, Set, Dict, Union, Optional
import warnings
import logging
from collections import Counter


class ContinuousBlockLabelProcessor:
    """
    A class for extracting and processing stimulus epochs from ECoG data.
    
    This class provides methods to identify stimulus boundaries, extract epochs
    with their preceding rest periods (or non preceding), and perform various analyses on the
    extracted data.
    """
        
    def __init__(self, 
                stimuli_codes: Set[Union[int, str]], 
                pre_stimuli_code: Optional[Union[int, str]] = None,
                pre_stimuli_duration: Optional[int] = None,
                stimuli_duration: Optional[int] = None, 
                truncate_longer_stimuli: bool = False,     
                ignore_codes: Optional[Set[Union[int, str]]] = None,
                log_level: int = logging.INFO):
        """
        Initialize the ECoG stimulus processor.
        
        Parameters:
        -----------
        stimuli_codes : Set[Union[int, str]]
            Set of codes representing stimuli to extract
        pre_stimuli_code : Optional[Union[int, str]], optional
            Code representing pre-stimulus rest periods (default: None)
            Set to 101 automatically if pre_stimuli_duration is provided
        pre_stimuli_duration : Optional[int], optional
            Duration (in timepoints) of pre-stimulus period to extract (default: None)
            If None, pre-stimulus processing will be disabled by default
        ignore_codes : Optional[Set[Union[int, str]]], optional
            Set of codes to ignore during processing (default: {0})
        log_level : int, optional
            Logging level (default: logging.INFO)
        """
        self.stimuli_codes = stimuli_codes
        self.pre_stimuli_code = pre_stimuli_code
        self.pre_stimuli_duration = pre_stimuli_duration
        self.pre_stimulus_available = pre_stimuli_duration is not None
        self.ignore_codes = ignore_codes if ignore_codes is not None else set()
        self.stimuli_duration = stimuli_duration
        self.truncate_longer_stimuli = truncate_longer_stimuli
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def get_stimuli_boundaries(self, array: np.ndarray) -> List[Tuple[Union[int, str], int, int]]:
        """
        Identify boundaries of consecutive stimuli in neural recording data.
        
        Parameters:
        -----------
        array : np.ndarray
            Input array containing stimuli markers
            
        Returns:
        --------
        List[Tuple[Union[int, str], int, int]]
            List of tuples (stimulus_value, start_index, end_index)
        """
        # Ensure array is a numpy array
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Flatten the array if it's multi-dimensional
        flattened = array.flatten()
        
        if len(flattened) == 0:
            return []
        
        stimuli_boundaries = []
        current_stimulus = flattened[0]
        start_index = 0
        
        # Iterate through the array
        for i in range(1, len(flattened)):
            if flattened[i] != current_stimulus:
                # Record the current run
                end_index = i 
                stimuli_boundaries.append((current_stimulus, start_index, end_index))
                current_stimulus = flattened[i]
                start_index = i
        
        # Add the final stimulus boundary
        stimuli_boundaries.append((current_stimulus, start_index, len(flattened)))
        
        return stimuli_boundaries
    
    
    def verify_stimulus_durations(self, 
                                boundaries: List[Tuple[Union[int, str], int, int]]) -> List[Tuple[Union[int, str], int, int]]:
        """
        Verify that all stimuli have the expected duration and normalize if allowed.
        
        Parameters:
        -----------
        boundaries : List[Tuple[Union[int, str], int, int]]
            List of boundary tuples (stimulus_value, start_index, end_index)
            
        Returns:
        --------
        List[Tuple[Union[int, str], int, int]]
            List of verified/normalized boundaries
        """
        if self.stimuli_duration is None:
            # No duration check requested, return as is
            return boundaries
        
        verified_boundaries = []
        
        for boundary in boundaries:
            stim_value, start_idx, end_idx = boundary
            
            # Only check stimuli that are in our target stimuli codes
            if stim_value in self.stimuli_codes:
                stim_duration = end_idx - start_idx
                
                if stim_duration < self.stimuli_duration:
                    # Stimulus is too short, always reject
                    self.logger.error(f"Stimulus {stim_value} at indices {start_idx}-{end_idx} has duration {stim_duration}, "
                                    f"which is shorter than expected {self.stimuli_duration}. Rejecting.")
                    continue
                    
                elif stim_duration > self.stimuli_duration:
                    # Stimulus is too long
                    if self.truncate_longer_stimuli:
                        # Truncate to expected duration
                        new_end_idx = start_idx + self.stimuli_duration
                        self.logger.warning(f"Stimulus {stim_value} at indices {start_idx}-{end_idx} has duration {stim_duration}, "
                                        f"truncating to first {self.stimuli_duration} timepoints.")
                        verified_boundaries.append((stim_value, start_idx, new_end_idx))
                    else:
                        # Reject if not allowing longer stimuli
                        self.logger.error(f"Stimulus {stim_value} at indices {start_idx}-{end_idx} has duration {stim_duration}, "
                                        f"which is longer than expected {self.stimuli_duration}. Rejecting.")
                        continue
                else:
                    # Duration is exactly as expected
                    verified_boundaries.append(boundary)
            else:
                # Not a target stimulus, keep as is
                verified_boundaries.append(boundary)
        
        return verified_boundaries    

    def prepend_pre_stimulus_periods(self, 
                                boundaries: List[Tuple[Union[int, str], int, int]]) -> List[Tuple[Union[int, str], int, int, int, int]]:
        """
        Extract stimulus epochs and prepend their preceding rest periods.
        Only returns stimuli that have valid pre-stimulus periods.
        
        Parameters:
        -----------
        boundaries : List[Tuple[Union[int, str], int, int]]
            Input boundaries as list of tuples (stimulus_value, start_index, end_index)
                
        Returns:
        --------
        List[Tuple[Union[int, str], int, int, int, int]]
            List of tuples (stimulus_value, pre_stim_start, pre_stim_end, stim_start, stim_end)
        """
        epochs = []
        
        for i, (stim_value, start_idx, end_idx) in enumerate(boundaries):
            # Skip if not in the stimuli codes we're looking for
            if stim_value not in self.stimuli_codes:
                continue
            
            # Check if this is the first boundary
            if i == 0:
                self.logger.error(f"Stimulus {stim_value} at {start_idx} cannot be the first boundary (no preceding rest period)")
                continue
            
            # Check if the previous boundary is a rest period
            if boundaries[i-1][0] == self.pre_stimuli_code:
                rest_start, rest_end = boundaries[i-1][1], boundaries[i-1][2]
                rest_duration = rest_end - rest_start
                
                # Check if rest period is long enough
                if rest_duration < self.pre_stimuli_duration:
                    self.logger.error(f"Rest period before stimulus {stim_value} at {start_idx} is too short: {rest_duration}, skipping this epoch")
                    continue
                
                # Calculate pre-stimulus period
                if rest_duration > self.pre_stimuli_duration:
                    pre_stim_start = rest_end - self.pre_stimuli_duration
                    self.logger.warning(
                        f"Rest period before stimulus {stim_value} at {start_idx} is longer than {self.pre_stimuli_duration} timepoints, "
                        f"using only the last {self.pre_stimuli_duration} timepoints for the pre-stimulus period"
                    )
                else:
                    pre_stim_start = rest_start
                
                # Add the epoch
                epochs.append((stim_value, pre_stim_start, end_idx))
                # epochs.append((stim_value, pre_stim_start, rest_end, start_idx, end_idx))
            else:
                self.logger.error(f"Stimulus {stim_value} at {start_idx} has no preceding rest period, skipping this epoch")
        
        return epochs
    
    def get_stimulus_boundaries_only(self, 
                               boundaries: List[Tuple[Union[int, str], int, int]]) -> List[Tuple[Union[int, str], int, int]]:
        """
        Extract just the stimulus boundaries without requiring pre-stimulus periods.
        
        Parameters:
        -----------
        boundaries : List[Tuple[Union[int, str], int, int]]
            Input boundaries as list of tuples (stimulus_value, start_index, end_index)
                
        Returns:
        --------
        List[Tuple[Union[int, str], int, int]]
            List of tuples (stimulus_value, start_index, end_index) for matching stimuli
        """
        return [(v, s, e) for v, s, e in boundaries if v in self.stimuli_codes]

    def check_for_unexpected_codes(self, 
                                 boundaries: List[Tuple[Union[int, str], int, int]]) -> List[Tuple[Union[int, str], int, int]]:
        """
        Check for unexpected codes in the boundaries.
        
        Parameters:
        -----------
        boundaries : List[Tuple[Union[int, str], int, int]]
            Input boundaries as list of tuples (stimulus_value, start_index, end_index)
            
        Returns:
        --------
        List[Tuple[Union[int, str], int, int]]
            List of boundaries with unexpected codes
        """
        unexpected_codes = []
        expected_codes = set.union(self.stimuli_codes, {self.pre_stimuli_code}, self.ignore_codes)
        
        for i, (stim_value, start_idx, end_idx) in enumerate(boundaries):
            if stim_value not in expected_codes:
                self.logger.warning(f"Found unexpected code {stim_value} at indices {start_idx}-{end_idx}")
                unexpected_codes.append((stim_value, start_idx, end_idx))
        
        return unexpected_codes
    
    def process_label_data(self, 
                    array: np.ndarray, 
                    check_unexpected_codes: bool = True,
                    include_pre_stimulus: Optional[bool] = None) -> Dict[str, Union[List, Dict]]:

        """
        Process ECoG data to extract stimulus epochs.
        
        Parameters:
        -----------
        array : np.ndarray
            Input array containing stimuli markers
        check_unexpected_codes : bool, optional
            Whether to check for unexpected codes (default: True)
        include_pre_stimulus : Optional[bool], optional
            Whether to include pre-stimulus periods (default: None)
            If None, uses the availability determined during initialization
            
        Returns:
        --------
        Dict
            Dictionary with processed results
        """
        # Determine if pre-stimulus should be included
        if include_pre_stimulus is None:
            include_pre_stimulus = self.pre_stimulus_available
        
        # Check if pre-stimulus is requested but not available
        if include_pre_stimulus and not self.pre_stimulus_available:
            self.logger.warning("Pre-stimulus processing requested but not configured. "
                            "Set pre_stimuli_duration during initialization.")
            include_pre_stimulus = False
        
        # Get boundaries
        boundaries = self.get_stimuli_boundaries(array)
        
        # Verify stimulus durations if requested
        if self.stimuli_duration is not None:
            boundaries = self.verify_stimulus_durations(boundaries)
            
        # Process according to whether pre-stimulus is needed
        if include_pre_stimulus:
            epochs = self.prepend_pre_stimulus_periods(boundaries)
        else:
            # Just get the stimulus boundaries without pre-stimulus periods
            epochs = self.get_stimulus_boundaries_only(boundaries)
        
        # Count stimuli
        stimulus_counts = Counter([epoch[0] for epoch in epochs])
        
        result = {
            'boundaries': boundaries,
            'epochs': epochs,
            'stimulus_counts': dict(stimulus_counts)
        }
        
        # Check for unexpected codes if requested
        if check_unexpected_codes:
            unexpected_codes = self.check_for_unexpected_codes(boundaries)
            result['unexpected_codes'] = unexpected_codes
            logging.info(f"Found {len(unexpected_codes)} unexpected codes")
        
        return result

    ##Below : other methods, that may be used
       
    ##get_epoch_data isn't tested yet!! (output of claude 3.7) ==> must test later if this is to be used 
    '''
    def get_epoch_data(self, 
                    data: np.ndarray, 
                    epochs: List[Tuple[Union[int, str], int, int]]) -> Dict[Union[int, str], List[np.ndarray]]:
        """
        Extract actual data for each epoch from the full data array.
        
        Parameters:
        -----------
        data : np.ndarray
            Full data array (can be multi-dimensional, e.g., channels × time)
        epochs : List[Tuple[Union[int, str], int, int]]
            List of epochs as (stimulus_value, start_index, end_index)
            
        Returns:
        --------
        Dict[Union[int, str], List[np.ndarray]]
            Dictionary mapping stimulus values to lists of epoch data arrays
        """
        epoch_data = {}
        
        for epoch in epochs:
            stim_value, start_idx, end_idx = epoch
            
            # Handle multi-dimensional data by taking all other dimensions as is
            if data.ndim > 1:
                # For multi-dimensional data, extract all dimensions at the specified time indices
                # Create slice objects for all dimensions, with time dimension sliced according to the epoch
                full_epoch_slices = tuple([slice(None)] * (data.ndim - 1) + [slice(start_idx, end_idx)])
                epoch_array = data[full_epoch_slices]
            else:
                # For one-dimensional data, simply slice the array
                epoch_array = data[start_idx:end_idx]
            
            # Add to the dictionary, creating a list for the stim_value if it doesn't exist
            if stim_value not in epoch_data:
                epoch_data[stim_value] = []
            
            epoch_data[stim_value].append(epoch_array)
        
        return epoch_data
    '''

