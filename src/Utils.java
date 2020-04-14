import java.io.FileReader;
import java.io.IOException;

public class Utils {
    static String readFile(String path) throws IOException {
        FileReader reader = new FileReader(path);
        String content = "";
        int c;
        while ((c=reader.read()) != -1) {
            content += (char) c;
        }
        return content;
    }
}
