import pandas as pd
import numpy as np
import chess
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import random


class ChessBot:
    """Chess bot using Random Forest machine learning model"""
    
    def __init__(self, csv_file='games.csv'):
        self.csv_file = csv_file
        # random forest with 50 trees for memory efficiency
        self.model = RandomForestClassifier(
            n_estimators=50, 
            max_depth=15,
            random_state=42,
            n_jobs=2  # limit parallel processing
        )
        self.move_encoder = LabelEncoder()  # converts moves to numbers
        self.is_trained = False
        self.max_games_to_train = 2000  # reduced for memory
        self.max_positions = 100000  # cap total training positions
        self.position_move_pairs = []
        
    def extract_features(self, board):
        """
        Extract numerical features from a chess board position
        Returns a feature vector representing the board state
        """
        features = []
        
        # count each piece type for both colors (12 features)
        for color in [chess.WHITE, chess.BLACK]:
            for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, 
                              chess.ROOK, chess.QUEEN, chess.KING]:
                count = len(board.pieces(piece_type, color))
                features.append(count)
        
        # piece distribution across ranks (16 features)
        for color in [chess.WHITE, chess.BLACK]:
            for rank in range(8):
                count = 0
                for file in range(8):
                    square = chess.square(file, rank)
                    piece = board.piece_at(square)
                    if piece and piece.color == color:
                        count += 1
                features.append(count)
        
        # center control for both sides (2 features)
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        for color in [chess.WHITE, chess.BLACK]:
            center_control = sum(1 for sq in center_squares 
                                if board.piece_at(sq) and board.piece_at(sq).color == color)
            features.append(center_control)
        
        # game state features (8 features)
        features.append(1 if board.turn == chess.WHITE else 0)  # whose turn
        features.append(1 if board.is_check() else 0)  # check status
        features.append(board.fullmove_number)  # move counter
        features.append(1 if board.has_kingside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.WHITE) else 0)
        features.append(1 if board.has_kingside_castling_rights(chess.BLACK) else 0)
        features.append(1 if board.has_queenside_castling_rights(chess.BLACK) else 0)
        features.append(len(list(board.legal_moves)))  # mobility
        
        return np.array(features)
    
    def load_and_train(self):
        """Load games from CSV and train the ML model"""
        print("Loading chess games dataset")
        
        # process CSV in chunks to avoid loading all into memory
        chunk_size = 500
        X_features = []  # feature vectors
        y_moves = []  # corresponding moves
        games_processed = 0
        
        print(f"Processing up to {self.max_games_to_train} games")
        print(f"Max positions: {self.max_positions}")
        
        # iterate through CSV in chunks
        for chunk in pd.read_csv(self.csv_file, chunksize=chunk_size):
            for idx, row in chunk.iterrows():
                if games_processed >= self.max_games_to_train:
                    break
                if len(X_features) >= self.max_positions:
                    break
                    
                try:
                    moves_str = row['moves']
                    if pd.notna(moves_str):
                        # extract features and moves from this game
                        features, moves = self._process_game_for_ml(moves_str)
                        X_features.extend(features)
                        y_moves.extend(moves)
                        games_processed += 1
                        
                        if games_processed % 200 == 0:
                            print(f"Processed {games_processed} games... ({len(X_features)} positions)")
                except Exception:
                    continue
            
            if games_processed >= self.max_games_to_train or len(X_features) >= self.max_positions:
                break
        
        if len(X_features) == 0:
            print("No training data extracted!")
            return
        
        # cap at max positions if needed
        if len(X_features) > self.max_positions:
            print(f"Limiting to {self.max_positions} positions to save memory")
            X_features = X_features[:self.max_positions]
            y_moves = y_moves[:self.max_positions]
        
        print(f"\nTraining Random Forest model on {len(X_features)} positions")
        
        # convert to numpy array (float32 saves memory)
        X = np.array(X_features, dtype=np.float32) 
        
        # encode moves as integer labels
        self.move_encoder.fit(y_moves)
        y = self.move_encoder.transform(y_moves)
        
        # free memory
        X_features = None
        y_moves = None
        
        self.model.fit(X, y)
        self.is_trained = True
        
        print(f"Training complete!")
        print(f"Model trained on {len(X)} positions")
        print(f"Vocabulary: {len(self.move_encoder.classes_)} unique moves")
        
    def _process_game_for_ml(self, moves_str):
        """Process a single game and extract feature-move pairs"""
        board = chess.Board()
        moves = moves_str.split()
        
        features_list = []
        moves_list = []
        
        max_moves_per_game = 40  # limit to save memory
        
        for i, move in enumerate(moves):
            if i >= max_moves_per_game:
                break
                
            try:
                # get position features
                features = self.extract_features(board)
                
                # parse move notation
                chess_move = board.parse_san(move)
                move_uci = chess_move.uci()
                
                # save this position-move pair
                features_list.append(features)
                moves_list.append(move_uci)
                
                # apply move to board
                board.push(chess_move)
                
            except Exception:
                break
        
        return features_list, moves_list
    
    def get_bot_move(self, board):
        """Get the bot's move using the trained ML model"""
        # fallback if not trained
        if not self.is_trained:
            return self._get_random_move(board)
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        # get position features
        features = self.extract_features(board).reshape(1, -1)
        
        try:
            # get move probabilities from model
            probabilities = self.model.predict_proba(features)[0]
            
            # sort by probability (highest first)
            top_indices = np.argsort(probabilities)[::-1]
            
            # try top 20 predicted moves
            for idx in top_indices[:20]:
                predicted_move_uci = self.move_encoder.classes_[idx]
                try:
                    predicted_move = chess.Move.from_uci(predicted_move_uci)
                    # return first legal move found
                    if predicted_move in legal_moves:
                        return predicted_move
                except:
                    continue
            
            # fallback to heuristics
            return self._get_heuristic_move(board)
            
        except Exception:
            return self._get_heuristic_move(board)
    
    def _get_random_move(self, board):
        """Get a random legal move"""
        legal_moves = list(board.legal_moves)
        return random.choice(legal_moves) if legal_moves else None
    
    def _get_heuristic_move(self, board):
        """Get a move using heuristic evaluation"""
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            return None
        
        # score all legal moves
        move_scores = []
        for move in legal_moves:
            score = self._evaluate_move(board, move)
            move_scores.append((move, score))
        
        # return highest scoring move
        move_scores.sort(key=lambda x: x[1], reverse=True)
        return move_scores[0][0]
    
    def _evaluate_move(self, board, move):
        """Simple heuristic evaluation of a move"""
        score = 0
        
        # test the move
        board.push(move)
        
        # checkmate is best
        if board.is_checkmate():
            score += 10000
        elif board.is_check():
            score += 50
        
        # undo the move
        board.pop()
        
        # value captures by piece worth
        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                piece_values = {
                    chess.PAWN: 100,
                    chess.KNIGHT: 320,
                    chess.BISHOP: 330,
                    chess.ROOK: 500,
                    chess.QUEEN: 900,
                    chess.KING: 0
                }
                score += piece_values.get(captured_piece.piece_type, 0)
        
        # reward center control
        center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        if move.to_square in center_squares:
            score += 30
        
        # add randomness for variety
        score += random.randint(-10, 10)
        
        return score
    
    def save_model(self, filename='chess_bot_ml_model.pkl'):
        """Save the trained ML model"""
        model_data = {
            'model': self.model,
            'move_encoder': self.move_encoder,
            'is_trained': self.is_trained
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"ML model saved to {filename}")
    
    def load_model(self, filename='chess_bot_ml_model.pkl'):
        """Load a trained ML model"""
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    model_data = pickle.load(f)
                self.model = model_data['model']
                self.move_encoder = model_data['move_encoder']
                self.is_trained = model_data['is_trained']
                print(f"ML model loaded from {filename}")
                print(f"Model knows {len(self.move_encoder.classes_)} unique moves")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        return False
