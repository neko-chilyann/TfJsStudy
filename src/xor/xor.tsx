import React, { Component } from "react";
import * as tf from '@tensorflow/tfjs';
import * as tfVis from '@tensorflow/tfjs-vis';
import { getData } from "./data";

/**
 * Xor
 *
 * @export
 * @class Xor
 * @extends {Component<any, any>}
 */
export default class Xor extends Component<any, any> {

    /**
     * tenserflow模型
     *
     * @protected
     * @memberof Xor
     */
    protected model!: tf.Sequential;

    /**
     * Creates an instance of Xor.
     * @param {*} props
     * @memberof Xor
     */
    constructor(props: any) {
        super(props);
        this.state = {
            x: 0,
            y: 0
        };
    }

    /**
     * 
     *
     * @returns {Promise<void>}
     * @memberof Xor
     */
    public async componentDidMount(): Promise<void> {
        this.init();
    }

    /**
     *
     *
     * @protected
     * @returns {Promise<void>}
     * @memberof Xor
     */
    protected async init(): Promise<void> {
        // 获取数据
        const data = getData(400);
        // 可视化展示数据
        tfVis.render.scatterplot(
            { name: 'Xor 数据集' },
            {
                values: [
                    data.filter(p => p.label === 1),
                    data.filter(p => p.label === 0),
                ]
            }
        );
        //转换数据集为tensor
        const inputs = tf.tensor(data.map(p => [p.x, p.y]));
        const labels = tf.tensor(data.map(p => p.label));
        // 初始化模型
        this.model = tf.sequential();
        // 添加全连接隐藏层
        this.model.add(tf.layers.dense({ units: 4, inputShape: [2], activation: 'relu' }));
        // 添加输出层
        this.model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
        // 对数损失函数、优化器
        this.model.compile({ loss: tf.losses.logLoss, optimizer: tf.train.adam(0.1) });
        // 训练
        await this.model.fit(inputs, labels, {
            epochs: 10,
            callbacks: tfVis.show.fitCallbacks(
                { name: '训练过程' },
                ['loss']
            )
        });
    }

    /**
     * 预测点击
     *
     * @protected
     * @memberof Xor
     */
    protected btnClick(): void {
        const output: any = this.model.predict(tf.tensor([[this.state.x * 1, this.state.y * 1]]));
        alert(output.dataSync());
    }

    /**
     * 值变更
     *
     * @protected
     * @memberof Xor
     */
    protected change(name: string, e: any): void {
        const data: any = {};
        data[name] = e.target.value;
        this.setState(data);
    }

    /**
     *
     *
     * @returns
     * @memberof Xor
     */
    public render() {
        return <div className="logistic-regression">
            x轴坐标：<input type="text" value={this.state.x} onChange={(e: any) => this.change('x', e)} />
            y轴坐标：<input type="text" value={this.state.y} onChange={(e: any) => this.change('y', e)} />
            <button onClick={() => this.btnClick()}>预测</button>
        </div>;
    }

}