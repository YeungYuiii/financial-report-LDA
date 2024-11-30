import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class DFReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    private IntWritable dfCount = new IntWritable();

    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
        int df = 0;
        for (IntWritable val : values) {
            df += val.get();
        }
        dfCount.set(df);
        context.write(key, dfCount);
    }
}
