import React, { Component } from "react";
import * as tf from '@tensorflow/tfjs';
import * as tfVis from '@tensorflow/tfjs-vis';
import { MnistData } from './data';

/**
 * 卷积神经网络学习
 *
 * @export
 * @class mnist
 * @extends {Component}
 */
export default class mnist extends Component<any, any> {
    /**
     * Sequential模型
     *
     * @protected
     * @type {tf.Sequential}
     * @memberof mnist
     */
    protected model!: tf.Sequential;
    /**
     * 数据获取器
     *
     * @protected
     * @type {MnistData}
     * @memberof mnist
     */
    protected data!: MnistData;
    /**
     * 获取画布
     *
     * @readonly
     * @protected
     * @memberof mnist
     */
    protected get canvas(): HTMLCanvasElement {
        const ref: any = this.refs.canvas;
        return ref;
    }

    /**
     * 组件加载完毕
     *
     * @memberof mnist
     */
    public componentDidMount() {
        this.init();
    }

    /**
     * 初始化
     *
     * @returns {Promise<void>}
     * @memberof mnist
     */
    public async init(): Promise<void> {
        // 数据集对象
        this.data = new MnistData();
        // 加载数据集
        await this.data.load();
        // 获取输入示例
        const examples = this.data.nextTestBatch(20);
        // 新建一个绘图区面板
        const surface = tfVis.visor().surface({ name: '卷积神经网络学习-输入数据' });
        // 分割图片
        for (let i = 0; i < 20; i++) {
            const imageTensor: any = tf.tidy(() => {
                return examples.xs.slice([i, 0], [1, 784]).reshape([28, 28, 1]);
            });
            // 创建canvas画布
            const canvas = document.createElement('canvas');
            canvas.width = 28;
            canvas.height = 26;
            canvas.style.margin = '4px';
            // 使用tf转换为图片
            await tf.browser.toPixels(imageTensor, canvas);
            // 挂载
            surface.drawArea.appendChild(canvas);
        }
        this.initCanvas();
        this.initModel();
        this.trainingModel();
    }

    /**
     * 定义模型(卷积神经网络)
     *
     * @returns {Promise<void>}
     * @memberof mnist
     */
    public async initModel(): Promise<void> {
        // 创建模型
        this.model = tf.sequential();
        // 添加卷积层
        this.model.add(tf.layers.conv2d({
            // 数据形状
            inputShape: [28, 28, 1],
            kernelSize: 5,
            filters: 8,
            strides: 1,
            // 激活函数
            activation: 'relu',
            // 卷积核初始化方法
            kernelInitializer: 'varianceScaling'
        }));
        // 添加最大池化层
        this.model.add(tf.layers.maxPool2d({
            poolSize: [2, 2],
            strides: [2, 2]
        }));
        // 卷积层
        this.model.add(tf.layers.conv2d({
            kernelSize: 5,
            filters: 16,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'varianceScaling'
        }));
        // 最大池化层
        this.model.add(tf.layers.maxPool2d({
            poolSize: [2, 2],
            strides: [2, 2]
        }));
        // 平铺(摊平)操作：将高维数据转换为一维数据，用于最后分类
        this.model.add(tf.layers.flatten());
        // 全连接层
        this.model.add(tf.layers.dense({ units: 10, activation: 'softmax', kernelInitializer: 'varianceScaling' }));
    }

    /**
     * 训练模型
     *
     * @protected
     * @returns {Promise<void>}
     * @memberof mnist
     */
    protected async trainingModel(): Promise<void> {
        this.model.compile({
            loss: 'categoricalCrossentropy',
            optimizer: tf.train.adam(),
            metrics: 'accuracy'
        });
        // 使用tidy优化内存释放
        // 获取训练集
        const [trainXs, trainYs] = tf.tidy(() => {
            const num: number = 1000;
            const d = this.data.nextTrainBatch(num);
            return [
                d.xs.reshape([num, 28, 28, 1]),
                d.labels
            ];
        });
        // 获取验证集
        const [testXs, testYs] = tf.tidy(() => {
            const num: number = 200;
            const d = this.data.nextTestBatch(num);
            return [
                d.xs.reshape([num, 28, 28, 1]),
                d.labels
            ];
        });
        // 训练
        await this.model.fit(trainXs, trainYs, {
            validationData: [testXs, testYs],
            epochs: 50,
            callbacks: tfVis.show.fitCallbacks(
                { name: '训练效果' },
                ['loss', 'val_loss', 'acc', 'val_acc'],
                { callbacks: ['onEpochEnd'] }
            )
        });
    }

    /**
     * 验证
     *
     * @protected
     * @returns {Promise<void>}
     * @memberof mnist
     */
    protected verification = async (): Promise<void> => {
        const input = tf.tidy(() => {
            // 转换图片大小
            return tf.image.resizeBilinear(
                // 将canvas转换为tensor数据
                tf.browser.fromPixels(this.canvas),
                [28, 28],
                true
            )
                // 转换黑白图片
                .slice([0, 0, 0], [28, 28, 1])
                // 归一化操作
                .toFloat()
                .div(255)
                // 更改形状
                .reshape([1, 28, 28, 1]);
        });
        const pred: any = this.model.predict(input);
        pred.argMax(1);
        alert(`预测结果为：${pred.dataSync()[0]}`);
    }

    /**
     * 初始化画布
     *
     * @protected
     * @memberof mnist
     */
    protected initCanvas(): void {
        this.clear();
        // 监控鼠标事件，绘制手写数字
        this.canvas.addEventListener('mousemove', (e: MouseEvent) => {
            if (e.buttons === 1) {
                // 获取画布上下文
                const ctx = this.canvas.getContext('2d');
                if (ctx) {
                    // 填充样式
                    ctx.fillStyle = 'rgb(255, 255, 255)';
                    ctx.fillRect(e.offsetX, e.offsetY, 20, 20);
                }
            }
        });
    }

    /**
     * 清除画布
     *
     * @protected
     * @memberof mnist
     */
    protected clear = (): void => {
        // 获取画布上下文
        const ctx = this.canvas.getContext('2d');
        if (ctx) {
            // 填充样式
            ctx.fillStyle = 'rgb(0, 0, 0)';
            // 设置一个矩形
            ctx.fillRect(0, 0, 300, 300);
        }
    }

    /**
     * 绘制内容
     *
     * @returns
     * @memberof mnist
     */
    public render() {
        return <div className="container">
            <canvas ref="canvas" width="300" height="300" style={{ border: '2px solid #666' }} /><br />
            <button onClick={this.clear}>清除</button>
            <button onClick={this.verification}>预测</button>
        </div>;
    }

}