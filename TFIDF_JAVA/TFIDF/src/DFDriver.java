import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class DFDriver {
    public static void main(String[] args) throws Exception {
        if (args.length != 2) {
            System.err.println("Usage: DFDriver <input path> <output path>");
            System.exit(-1);
        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Document Frequency");
        job.setJarByClass(DFDriver.class);

        // Set custom InputFormat
        job.setInputFormatClass(WholeFileInputFormat.class);

        // Set Mapper and Reducer classes
        job.setMapperClass(DFMapper.class);
        job.setReducerClass(DFReducer.class);

        // Set Mapper output key/value types
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        // Set final output key/value types
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        // Set input and output paths
        WholeFileInputFormat.addInputPath(job, new Path(args[0]));
        TextOutputFormat.setOutputPath(job, new Path(args[1]));

        // Optionally set Combiner
        job.setCombinerClass(DFReducer.class);

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
