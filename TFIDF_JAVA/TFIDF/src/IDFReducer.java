import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import java.io.IOException;

public class IDFReducer extends Reducer<Text, Text, Text, Text> {
    private double totalDocs;

    @Override
    protected void setup(Context context) throws IOException, InterruptedException {
        Configuration conf = context.getConfiguration();
        totalDocs = conf.getDouble("totalDocs", 1.0);
    }

    @Override
    protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
        for (Text val : values) {
            String[] parts = val.toString().split("\t");
            if (parts.length != 2) continue;

            String word = parts[0];
            int df = Integer.parseInt(parts[1]);

            if (df == 0) continue; // Avoid division by zero

            double idf = Math.log((double) totalDocs / (double) df);
            context.write(new Text(word), new Text(String.valueOf(idf)));
        }
    }
}
