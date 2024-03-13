dict_config = {
    "version": 1,
    "disable_exiting_loggers": False,
    "formatters": {
        "base": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(filename)s | %(funcName)s | %(message)s",
            "datefmt": "%d.%m.%Y %H:%M:%S"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "base"
        },
        "api_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "base",
            "filename": "practice/logs/api.log",
            "when": "h",
            "interval": 24,
            "backupCount": 3
        },
        "cpc_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "base",
            "filename": "practice/logs/cpc.log",
            "when": "h",
            "interval": 24,
            "backupCount": 3
        },
	"creative_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "base",
            "filename": "practice/logs/creatives.log",
            "when": "h",
            "interval": 24,
            "backupCount": 3
        },
	"tools_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "base",
            "filename": "practice/logs/tools.log",
            "when": "h",
            "interval": 24,
            "backupCount": 3
        },
	"train_file": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "level": "INFO",
            "formatter": "base",
            "filename": "practice/logs/train.log",
            "when": "h",
            "interval": 24,
            "backupCount": 3
        },



    },
    "loggers": {
      "api_logger": {
          "level": "INFO",
          "handlers": ["api_file"],
          #"force": True
      },
      "cpc_logger": {
          "level": "INFO",
          "handlers": ["cpc_file"],
      },
      "creative_logger": {
          "level": "INFO",
          "handlers": ["creative_file"],
      },
      "tools_logger": {
          "level": "INFO",
          "handlers": ["tools_file"],
      },
      "train_logger": {
          "level": "INFO",
          "handlers": ["train_file"],
      },




    },

}