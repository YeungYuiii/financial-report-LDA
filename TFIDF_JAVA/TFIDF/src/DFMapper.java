import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;
import java.util.HashSet;
import java.util.regex.Pattern;

public class DFMapper extends Mapper<NullWritable, Text, Text, IntWritable> {
    private static final IntWritable ONE = new IntWritable(1);
    private Text word = new Text();
    private HashSet<String> uniqueWords = new HashSet<>();
    private static final Pattern WORD_PATTERN = Pattern.compile("\\W+");

    @Override
    protected void map(NullWritable key, Text value, Context context) throws IOException, InterruptedException {
        // Retrieve document content
        String document = value.toString();

        // Clear the HashSet for the new document
        uniqueWords.clear();

        // Split document into words using pre-compiled Pattern
        String[] words = WORD_PATTERN.split(document);
        for (String w : words) {
            if (!w.isEmpty()) {
                uniqueWords.add(w.toLowerCase());
            }
        }

        // Emit each unique word with a count of 1
        for (String uniqueWord : uniqueWords) {
            word.set(uniqueWord);
            context.write(word, ONE);
        }
    }
}
