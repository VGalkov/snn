package org.example;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.UnaryOperator;

public class Visualiser extends JFrame implements Runnable, MouseListener {

    private final int w = 1280;
    private final int h = 720;
    private final BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
    private final BufferedImage pimg = new BufferedImage(w / 8, h / 8, BufferedImage.TYPE_INT_RGB);
    private final NN_Algorithm nn;
    public List<Point> points = new ArrayList<>();
    private final Color POINT_TYPE_LCM = Color.GREEN;
    private final Color POINT_TYPE_RCM = Color.BLUE;

    public Visualiser() {
        UnaryOperator<Double> sigmoid = x -> 1 / (1 + Math.exp(-x));
        UnaryOperator<Double> dsigmoid = y -> y * (1 - y);

        nn = new NN_Algorithm(0.01, sigmoid, dsigmoid, 2, 5, 5, 2);

        this.setSize(w, h);
        this.setVisible(true);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setLocation(50, 50);
        this.add(new JLabel(new ImageIcon(img)));
        addMouseListener(this);
    }

    @Override
    public void run() {
        while (true) {
            this.repaint();
            try {
                Thread.sleep(10);
            } catch (InterruptedException e) {
                System.out.println(Arrays.toString(e.getStackTrace()));
            }
        }
    }

    @Override
    public void paint(Graphics g) {
        if (points.size() > 0) {
            for (int k = 0; k < points.size() * 100; k++) {
                Point p = points.get((int) (Math.random() * points.size()));
                double nx = (double) p.x / w - 0.5;
                double ny = (double) p.y / h - 0.5;
                double[] targets = new double[2];

                if (POINT_TYPE_LCM.equals(p.type))
                    targets[0] = 1;
                else
                    targets[1] = 1;

                nn.feedForward(new double[]{nx, ny});
                nn.backpropagation(targets);
            }
        }

        for (int i = 0; i < w / 8; i++)
            for (int j = 0; j < h / 8; j++)
                pimg.setRGB(i, j, getRGB(i, j));

        Graphics ig = img.getGraphics();
        ig.drawImage(pimg, 0, 0, w, h, this);
        drawPoints(ig);
        g.drawImage(img, 8, 30, w, h, this);
    }

    private void drawPoints(Graphics ig) {
        for (Point p : points) {
            ig.setColor(Color.WHITE);
            ig.fillOval(p.x - 3, p.y - 3, 26, 26);
            ig.setColor(p.type);
            ig.fillOval(p.x, p.y, 20, 20);
        }
    }

    private int getRGB(int i, int j) {
        double[] outputs = nn.feedForward(new double[]{(double) i / w * 8 - 0.5, (double) j / h * 8 - 0.5});
        double green = 0.3 + Math.max(0, Math.min(1, outputs[0] - outputs[1] + 0.5)) * 0.5;
        double blue = 0.5 + (1 - green)* 0.5;

        return (100 << 16) | ((int) (green * 255) << 8) | (int) (blue * 255);
    }

    @Override
    public void mouseClicked(MouseEvent e) {

    }

    @Override
    public void mousePressed(MouseEvent e) {
        points.add(new Point(e.getX() - 16, e.getY() - 38, e.getButton() == 3 ? POINT_TYPE_LCM : POINT_TYPE_RCM));
    }

    @Override
    public void mouseReleased(MouseEvent e) {

    }

    @Override
    public void mouseEntered(MouseEvent e) {

    }

    @Override
    public void mouseExited(MouseEvent e) {

    }

    private static class Point {

        public int x;
        public int y;
        public Color type;

        public Point(int x, int y, Color type) {
            this.x = x;
            this.y = y;
            this.type = type;
        }
    }
}