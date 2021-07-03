#!/bin/bash
export PYTHONPATH=$PYTHONPATH:project_name  # add your project folder to python path
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
      python project_name/main.py
      ;;
    eval)
      python project_name/main.py
      ;;
    run)
      python project_name/model/dataloader.py $2
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
