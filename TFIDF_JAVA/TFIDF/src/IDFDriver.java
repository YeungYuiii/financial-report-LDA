import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat; // Added import
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat; // Added import
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

public class IDFDriver {
    public static void main(String[] args) throws Exception {
        if (args.length != 3) {
            System.err.println("Usage: IDFDriver <df input path> <totalDocs> <output path>");
            System.exit(-1);
        }

        String dfInputPath = args[0];
        double totalDocs = Double.parseDouble(args[1]);
        String outputPath = args[2];

        Configuration conf = new Configuration();
        conf.setDouble("totalDocs", totalDocs);

        Job job = Job.getInstance(conf, "Inverse Document Frequency");
        job.setJarByClass(IDFDriver.class);

        // Set Mapper and Reducer classes
        job.setMapperClass(IDFMapper.class);
        job.setReducerClass(IDFReducer.class);

        // Set output key/value types
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        // Set input and output paths
        FileInputFormat.addInputPath(job, new Path(dfInputPath)); // Corrected InputFormat
        TextOutputFormat.setOutputPath(job, new Path(outputPath));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
