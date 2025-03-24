def player(prev_play, opponent_moves=[], sequence_counts={}):
    """
    Rock Paper Scissors AI Player using Pattern Recognition and Predictive Modeling

    This function implements a simple machine learning strategy for Rock Paper Scissors:
    - Uses a short-term sequence prediction technique
    - Employs a form of n-gram frequency analysis
    - Implements a basic predictive model based on move history

    Key Machine Learning Concepts:
    - Pattern Recognition: Identifies recurring move sequences
    - Frequency Analysis: Tracks the most common move patterns
    - Markov Chain-like Prediction: Predicts next move based on previous move sequences
    - Adaptive Learning: Dynamically updates move prediction as more data is collected

    Strategy:
    1. Starts with a default prediction of Paper
    2. Tracks sequences of last 5 moves
    3. Identifies the most frequent subsequent move
    4. Counters the predicted move with its weakness

    Limitations:
    - Short memory (only considers last 5 moves)
    - Simple predictive model
    - No deep learning or complex neural network approach
    """
    # Default first move
    if not prev_play:
        prev_play = 'R'

    # Track opponent's move history
    opponent_moves.append(prev_play)

    # Default prediction (will get overwritten if enough data)
    predicted_move = 'P'

    # Start prediction only after we have at least 5 previous moves
    if len(opponent_moves) > 4:
        # Build the key: last 5 opponent moves as a string
        recent_sequence = "".join(opponent_moves[-5:])
        # Count how often this exact sequence has occurred
        sequence_counts[recent_sequence] = sequence_counts.get(recent_sequence, 0) + 1

        # Build possible 6-move sequences by adding R, P, or S to the last 4
        possible_next_sequences = [
            "".join([*opponent_moves[-4:], next_move]) 
            for next_move in ['R', 'P', 'S']
        ]

        # Check how many times each of the possible next sequences has occurred
        next_sequence_counts = {
            seq: sequence_counts[seq]
            for seq in possible_next_sequences if seq in sequence_counts
        }

        # Predict the most likely next move based on the highest frequency
        if next_sequence_counts:
            most_likely_sequence = max(next_sequence_counts, key=next_sequence_counts.get)
            predicted_move = most_likely_sequence[-1]  # The predicted next opponent move

    # Choose the move that beats the predicted opponent move
    counter_move = {'P': 'S', 'R': 'P', 'S': 'R'}
    return counter_move[predicted_move]
