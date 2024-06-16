from Parameters import Parameters
import numpy as np
from datetime import datetime


class Data_preparer():
    
    def __init__(self):
        self.data = None
        self.data_for_frame_1 = None
        self.data_for_frame_2 = None
        self.data_for_frame_3 = None
        self.fist_candle_1 = None
        self.fist_candle_2 = None
        self.fist_candle_3 = None
        self.training_sets = None
        self.set_for_frame_1 = None
        self.set_for_frame_1 = None
        self.set_for_frame_1 = None
        

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Data_preparer, cls).__new__(cls)
        return cls.instance
    
    def get_fist_candles(self, number_of_piece):
        return self.fist_candle_1[number_of_piece], self.fist_candle_2[number_of_piece], self.fist_candle_3[number_of_piece]
    
    def size_of_piece(self):
        return int(self.data.shape[0] / Parameters.number_of_pieces)
    
    def set_data(self):
        self.data = np.loadtxt(Parameters.data_name, delimiter=",") 

    """ возвращает все данные """ 
    def get_data(self):
        return self.data

    def get_piece(self, i):
        start = i * self.size_of_piece()
        end = (i + 1) * self.size_of_piece()
        return self.data[start:end, :]
     
    """ создает данные для одного фрейма для всех данных"""    
    def create_data_for_frame(self, base_frame, frame):
        
        def get_first_candle_for_frame (data, frame):
            
            for i, timestamp in enumerate(data[:,0]):
                date_time = datetime.fromtimestamp(timestamp / 1000)
                current_hour = date_time.hour
                current_minute = date_time.minute        
                current_minute_all = current_minute + current_hour * 60
        
                if current_minute_all % frame == 0:
                    return i
            return -1
        
        length_data = len(self.data)
        
        data_for_frame = np.zeros((Parameters.number_of_pieces,self.size_of_piece()//int(frame/base_frame)+2, 6));
        first_candles_for_frame = np.zeros(Parameters.number_of_pieces)
        ratio = int(frame/base_frame)
        
        for piece in range(Parameters.number_of_pieces):
            data = self.get_piece(piece)

            first_candle = get_first_candle_for_frame(data, frame)
            first_candles_for_frame[piece] = first_candle
            j = 0  
            if (first_candle > 0):
                data_for_frame[piece, j, 0] = data[0, 0]
                data_for_frame[piece, j, 1] = data[0, 1]
                data_for_frame[piece, j, 2] = np.max(data[0:first_candle,2])
                data_for_frame[piece, j, 3] = np.min(data[0:first_candle,3])
                data_for_frame[piece, j, 4] = data[first_candle - 1, 4]
                data_for_frame[piece, j, 5] = np.sum(data[0:first_candle, 5])
                j += 1
            
            for i in range (first_candle + ratio - 1, len(data), int(frame/base_frame)):
                data_for_frame[piece, j, 0] = data[i - (ratio-1), 0]
                data_for_frame[piece, j, 1] = data[i - (ratio-1), 1]
                data_for_frame[piece, j, 2] = np.max(data[(i-(ratio-1)):(i+1),2])
                data_for_frame[piece, j, 3] = np.min(data[(i-(ratio-1)):(i+1),3])
                data_for_frame[piece, j, 4] = data[i, 4]
                data_for_frame[piece, j, 5] = np.sum(data[(i-(ratio-1)):(i+1), 5])
                j += 1
    
            if (i < len(data) - 1):
                data_for_frame[piece, j, 0] = data[i + 1, 0]
                data_for_frame[piece, j, 1] = data[i + 1, 1]
                data_for_frame[piece, j, 2] = np.max(data[(i+1):(len(data)),2])
                data_for_frame[piece, j, 3] = np.min(data[(i+1):(len(data)),3])
                data_for_frame[piece, j, 4] = data[len(data) - 1, 4]
                data_for_frame[piece, j, 5] = np.sum(data[(i+1):(len(data)), 5])
        
        return data_for_frame, first_candles_for_frame   
        
    """ создает данные для фреймов для всех данных"""    
    def set_data_for_frames(self, bot):
        self.data_for_frame_1, self.fist_candle_1 = self.create_data_for_frame(bot.get_frames()[0], bot.get_frames()[1])
        self.data_for_frame_2, self.fist_candle_2 = self.create_data_for_frame(bot.get_frames()[0], bot.get_frames()[2])
        self.data_for_frame_3, self.fist_candle_3 = self.create_data_for_frame(bot.get_frames()[0], bot.get_frames()[3])   
        
        
    def get_piece_for_frames(self, i):
        return self.data_for_frame_1[i], self.data_for_frame_2[i], self.data_for_frame_3[i]






    """ раздел для training_set"""

    """ создает набор обучения""" 
    def create_training_set(self):      
        if len(self.data) < Parameters.NUMBER_OF_CANDLES:
            print("Number of candles is less than " + Parameters.NUMBER_OF_CANDLES)
            return(-1)
        
        number_of_sets = len(self.data) - (Parameters.NUMBER_OF_CANDLES - 1)
        
        self.training_sets = np.zeros((number_of_sets, Parameters.NUMBER_OF_CANDLES, len(self.data[0])))
        
        for set_ in range(number_of_sets):
            self.training_sets[set_,:,:] = self.data[set_:Parameters.NUMBER_OF_CANDLES+set_,:]

    """ возвращает весь набор""" 
    def get_trainig_set(self):    
        return self.training_sets

    """ возвращает все наборы для одного куска """
    def get_piece_of_training_set(self, i):
        start = i * self.size_of_piece()
        end = (i + 1) * self.size_of_piece() - (Parameters.NUMBER_OF_CANDLES - 1)
        return self.training_sets[start:end, :,:]
    
    """ возвращает все наборы для одного куска только открытие для НС""" 
    def get_piece_of_training_set_open(self, i):
        start = i * self.size_of_piece()
        end = (i + 1) * self.size_of_piece() - (Parameters.NUMBER_OF_CANDLES - 1)
        return self.training_sets[start:end, -500:,1]
    
    """ возвращает один сет из набора""" 
    def get_one_training_set(self, number_of_set, piece):      
        return self.training_sets[number_of_set + piece * self.size_of_piece(), :,:]   
    
    """ создает данные для одного фрейма для набора обучения"""    
    def create_training_set_for_frame(self, base_frame, frame):
        
        def get_first_candle_for_frame (data, frame):
            
            for i, timestamp in enumerate(data[:,0]):
                date_time = datetime.fromtimestamp(timestamp / 1000)
                current_hour = date_time.hour
                current_minute = date_time.minute        
                current_minute_all = current_minute + current_hour * 60
        
                if current_minute_all % frame == 0:
                    return i
            return -1
        
        set_for_frame = np.zeros((len(self.training_sets), len(self.training_sets[0])//int(frame/base_frame)+2, 6));
        ratio = int(frame/base_frame)
        
        for set_ in range(len(set_for_frame)):
            data = self.training_sets[set_]

            first_candle = get_first_candle_for_frame(data, frame)
            j = 0  
            if (first_candle > 0):
                set_for_frame[set_, j, 0] = data[0, 0]
                set_for_frame[set_, j, 1] = data[0, 1]
                set_for_frame[set_, j, 2] = np.max(data[0:first_candle,2])
                set_for_frame[set_, j, 3] = np.min(data[0:first_candle,3])
                set_for_frame[set_, j, 4] = data[first_candle - 1, 4]
                set_for_frame[set_, j, 5] = np.sum(data[0:first_candle, 5])
                j += 1
            
            for i in range (first_candle + ratio - 1, len(data), int(frame/base_frame)):
                set_for_frame[set_, j, 0] = data[i - (ratio-1), 0]
                set_for_frame[set_, j, 1] = data[i - (ratio-1), 1]
                set_for_frame[set_, j, 2] = np.max(data[(i-(ratio-1)):(i+1),2])
                set_for_frame[set_, j, 3] = np.min(data[(i-(ratio-1)):(i+1),3])
                set_for_frame[set_, j, 4] = data[i, 4]
                set_for_frame[set_, j, 5] = np.sum(data[(i-(ratio-1)):(i+1), 5])
                j += 1
    
            if (i < len(data) - 1):
                set_for_frame[set_, j, 0] = data[i + 1, 0]
                set_for_frame[set_, j, 1] = data[i + 1, 1]
                set_for_frame[set_, j, 2] = np.max(data[(i+1):(len(data)),2])
                set_for_frame[set_, j, 3] = np.min(data[(i+1):(len(data)),3])
                set_for_frame[set_, j, 4] = data[len(data) - 1, 4]
                set_for_frame[set_, j, 5] = np.sum(data[(i+1):(len(data)), 5])
    
        return set_for_frame
    
    """ создает данные для фреймов для набора обучения"""    
    def set_training_set_for_frames(self, bot):
        self.set_for_frame_1 = self.create_training_set_for_frame(bot.get_frames()[0], bot.get_frames()[1])
        self.set_for_frame_2 = self.create_training_set_for_frame(bot.get_frames()[0], bot.get_frames()[2])
        self.set_for_frame_3 = self.create_training_set_for_frame(bot.get_frames()[0], bot.get_frames()[3])
  
    """ возвращает один сет из набора для всех вреймов"""
    def get_one_training_set_for_frames(self, number_of_set, piece):  
        return (self.set_for_frame_1[number_of_set + piece * self.size_of_piece(), :,:],
                self.set_for_frame_2[number_of_set + piece * self.size_of_piece(), :,:],
                self.set_for_frame_3[number_of_set + piece * self.size_of_piece(), :,:])
   
    """ возвращает все данные для одного куска для набора""" 
    def get_piece_of_training_sets_for_frames(self, i):
       start = i * self.size_of_piece()
       end = (i + 1) * self.size_of_piece() - (Parameters.NUMBER_OF_CANDLES - 1)
       return (self.set_for_frame_1[start:end, :,:],
               self.set_for_frame_2[start:end, :,:],
               self.set_for_frame_3[start:end, :,:])
    
   