pgrep -f "server_model" | while read pid
do 
echo $pid
kill -9 $pid
done
