import pandas as pd
import numpy as np
from IPython.display import display

def display_table_fixed(current_state, action, next_state, reward, done):
    """
    A fixed version of the display_table function that works with newer NumPy versions
    and handles different state representations.
    """
    # Extract arrays if current_state is a tuple
    if isinstance(current_state, tuple) and len(current_state) == 2:
        current_state = current_state[0]  # Extract the array part
    
    # Extract arrays if next_state is a tuple
    if isinstance(next_state, tuple) and len(next_state) == 2:
        next_state = next_state[0]  # Extract the array part
    
    # Convert to numpy arrays if they aren't already
    current_state = np.array(current_state)
    next_state = np.array(next_state)
    
    # Column names
    STATE_VECTOR_COL_NAME = 'State Vector'
    DERIVED_COL_NAME = 'Derived from the State Vector (the closer to zero, the better)'
    
    # Add derived information
    def add_derived_info(state):
        return np.hstack([
            state, 
            [(state[0]**2 + state[1]**2)**.5],
            [(state[2]**2 + state[3]**2)**.5],
            [np.abs(state[4])]
        ])
    
    modified_current_state = add_derived_info(current_state)
    modified_next_state = add_derived_info(next_state)
    
    states = np.vstack([
        modified_current_state, 
        modified_next_state,
        modified_next_state - modified_current_state,        
    ]).T
    
    # Get the values from states array for each row
    def get_state(i, dtype=None):
        values = states[i]
        if dtype is not None:
            values = values.astype(dtype)
        return {
            'Current State': values[0],
            'Next State': values[1],
            'Action': '',
            'Reward': '',
            'Episode Terminated': '',
        }
    
    # Actions
    action_labels = [
        "Do nothing",
        "Fire right engine",
        "Fire main engine",
        "Fire left engine",
    ]
    
    # Create the DataFrame
    df = pd.DataFrame({
        ('', '', ''): {'Action': action_labels[action], 'Reward': reward, 'Episode Terminated': done},
        (STATE_VECTOR_COL_NAME, 'Coordinate', 'X (Horizontal)'): get_state(0),
        (STATE_VECTOR_COL_NAME, 'Coordinate', 'Y (Vertical)'): get_state(1),
        (STATE_VECTOR_COL_NAME, 'Velocity', 'X (Horizontal)'): get_state(2),
        (STATE_VECTOR_COL_NAME, 'Velocity', 'Y (Vertical)'): get_state(3),
        (STATE_VECTOR_COL_NAME, 'Tilting', 'Angle'): get_state(4),
        (STATE_VECTOR_COL_NAME, 'Tilting', 'Angular Velocity'): get_state(5),
        (STATE_VECTOR_COL_NAME, 'Ground contact', 'Left Leg?'): get_state(6, bool),  # Using bool instead of np.bool
        (STATE_VECTOR_COL_NAME, 'Ground contact', 'Right Leg?'): get_state(7, bool),  # Using bool instead of np.bool
        (DERIVED_COL_NAME, 'Distance from landing pad', ''): get_state(8),
        (DERIVED_COL_NAME, 'Velocity', ''): get_state(9),
        (DERIVED_COL_NAME, 'Tilting Angle (absolute value)', ''): get_state(10),
    })
    
    # Style the table
    styled_df = (df
        .fillna('')
        .reindex(['Current State', 'Action', 'Next State', 'Reward', 'Episode Terminated'])
        .style
        .applymap(lambda x: 'background-color : grey' if x == '' else '')
        .set_table_styles([
            {"selector": "th", "props": [("border", "1px solid grey"), ('text-align', 'center')]},
            {"selector": "tbody td", "props": [("border", "1px solid grey"), ('text-align', 'center')]},
        ])
    )
    
    # Display the table
    display(styled_df)
    
    return styled_df

# Usage example:
# display_table_fixed(current_state, action, next_state, reward, done)