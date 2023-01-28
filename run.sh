#! /usr/bin/env bash

export PROJECT_NAME=project_name  # add your project folder to python path
export PYTHONPATH=$PYTHONPATH:$PROJECT_NAME
export COMET_LOGGING_CONSOLE=info

Help()
{
   # Display Help
   echo 
   echo "Facilitates running different stages of training and evaluation."
   echo 
   echo "options:"
   echo "train                      Starts training."
   echo "eval                       Starts evaluation."
   echo "run file_name              Runs file_name.py file."
   echo
}

run () {
  case $1 in
    train)
      python $PROJECT_NAME/main.py
      ;;
    eval)
      python $PROJECT_NAME/main.py
      ;;
    run)
      python $2
      ;;
    -h) # display Help
      Help
      exit
      ;;
    *)
      echo "Unknown '$1' argument. Please run with '-h' argument to see more details."
      # Help
      exit
      ;;
  esac
}

run $1 $2

echo "Done."
